import torch
import sys
import os
import dill
import shutil
from collections import OrderedDict
from datetime import datetime
from pytz import timezone
import numpy as np
import time

from config import config, MetaConfig
from probs import TimeStepsDist
from common import load_val_data, create_logger, generate_fixed_weights, halting_step_stats
from common import OPTIMIZER_DICT
from plots import plot_image_map_data, plot_qt_probs, loss_plot, plot_dist_optimization_steps, plot_gradient_stats
from plots import plot_actsb_qts, plot_image_map_losses, plot_halting_step_stats_with_loss, plot_loss_versus_halting_step
from plots import create_exper_label
import utils.batch_handler
import utils.validation_handler
from val_optimizer import validate_optimizer
from mnist_data_obj import MNISTDataSet


class Experiment(object):

    def __init__(self, run_args, config, set_seed=False):

        # during "stand-alone" evaluation we're creating Experiment objects and we need to make sure especially in case
        # of the MLP experiment with the MNIST dataset that the batches are equivalent. Hence we can "force" a seed here
        # Remember, Experiment object has MNIST dataset from pytorch attached as attribute.
        if set_seed:
            SEED = 2345
            torch.manual_seed(SEED)
            if run_args.cuda:
                torch.cuda.manual_seed(SEED)
            np.random.seed(SEED)

        self.args = run_args
        self.epoch_stats = {"loss": [], "param_error": [], "act_loss": [], "qt_hist": OrderedDict(), "opt_step_hist": {},
                            "opt_loss": [], "step_losses": OrderedDict(), "halting_step": OrderedDict(),
                            "halting_stats": OrderedDict(),
                            "kl_term": np.zeros(run_args.max_epoch), "penalty_term": np.zeros(run_args.max_epoch),
                            "weight_regularizer": np.zeros(run_args.max_epoch),
                            "grad_stats": np.zeros((run_args.max_epoch, 2)),  # store mean and stddev of gradients model
                            "duration": [], "step_acc": OrderedDict()}
        self.val_stats = {"loss": [], "param_error": [], "act_loss": [], "qt_hist": OrderedDict(), "opt_step_hist": {},
                          "step_losses": OrderedDict(), "step_loss_var": OrderedDict(), "opt_loss": [],
                          "step_param_losses": OrderedDict(),
                          "ll_loss": {}, "kl_div": {}, "kl_entropy": {}, "qt_funcs": OrderedDict(),
                          "loss_funcs": [] if (run_args.learner[0:6] != "act_sb" and run_args.learner != "meta_act") else {},
                          "halting_step": OrderedDict(), "kl_term": [], "penalty_term": [],
                          "duration": [], "halt_step_funcs": {}, "mlp_halt_stats": OrderedDict(),
                          "step_acc": OrderedDict()}
        self.epoch = 0
        self.output_dir = None
        self.model_path = None
        self.model_name = None
        # self.opt_step_hist = None
        self.avg_num_opt_steps = 0
        self.val_avg_num_opt_steps = 0
        self.config = config
        self.val_funcs = None
        self.past_epochs = 0
        self.max_time_steps = self.args.optimizer_steps
        self.run_validation = True if self.args.eval_freq != 0 else False
        self.pt_dist = TimeStepsDist(T=self.config.T, q_prob=self.config.pT_shape_param)
        self.meta_logger = None
        self.fixed_weights = None
        self.type_prior = "geometric"
        if run_args.learner == "meta_act":
            self.annealing_schedule = np.empty(self.args.max_epoch)
            self.annealing_schedule.fill(self.config.tau)
        elif run_args.learner == "act_sb" and run_args.version == "V3.2":
            self.annealing_schedule = np.empty(self.args.max_epoch)
            self.annealing_schedule.fill(self.config.kl_anneal_perc)
        else:
            self.annealing_schedule = np.ones(self.args.max_epoch)
        if run_args.learner == "meta" and run_args.version == "V7":
            # metaV7 uses incremental learning schedule
            self.generate_curriculum_regime()
        else:
            self.inc_learning_schedule = None
        self.batch_handler_class = None
        self.optimizer = None
        # when optimizing MLP we need the MNIST data set
        self.validation_handler_class = None
        if run_args.problem == "mlp":
            self.dta_set = MNISTDataSet(run_args.batch_size, use_cuda=run_args.cuda)
        else:
            self.dta_set = None
        self.training_horizon = None
        # learning rate decay variables
        if hasattr(self.config, "loss_threshold_lr_decay"):
            self.loss_threshold_lr_decay = self.config.loss_threshold_lr_decay
        else:
            self.loss_threshold_lr_decay = 0
        self.lr_decay_last_epoch = 0
        self.lr_decay_rate = 0.5
        self.learning_rates = []
        self.learning_rates.append(run_args.lr)
        # only used for mixed mlp experiments to alter between 1 and 2-layer MLPs
        self.binary_switch = 0
        # backward compatibility
        if not hasattr(self.args, "trunc_bptt"):
            self.args.__dict__["trunc_bptt"] = False

    def check_lr_decay(self, meta_optimizer, current_loss=None, decay_type="lr_step_decay"):
        do_lr_decay = False
        if decay_type == "lr_step_decay":
            if self.epoch - self.lr_decay_last_epoch == self.args.lr_step_decay - 1 \
                    or self.lr_decay_last_epoch == 0:
                do_lr_decay = True
                # first set epoch in which we decay the lr (note, we check at the end of an epoch, so we add one
                self.lr_decay_last_epoch = self.epoch + 1  # we decay for the next epoch
            else:
                do_lr_decay = False
        elif decay_type == "compare_val_loss":
            if len(self.val_stats["opt_loss"]) > 2:
                # did our validation loss increase?
                if self.val_stats["opt_loss"][-2] < self.val_stats["opt_loss"][-1]:
                    perc_diff = (self.val_stats["opt_loss"][-1] - self.val_stats["opt_loss"][-2]) \
                                  / float(self.val_stats["opt_loss"][-2])
                    # increase is more than 5% w.r.t. previous (lower) validation loss
                    self.meta_logger.info("!!! Learning rate decay: increased validation loss {:.2f}%".format(perc_diff*100))
                    if perc_diff >= 0.05:
                        do_lr_decay = True
        else:
            raise ValueError("Learning rate decay type {} is not supported".format(decay_type))

        if do_lr_decay:
            new_lr = self.lr_decay_rate * self.learning_rates[-1]  # multiply with the last lr we used
            # we need to construct the optimizer again with the model parameters
            self.optimizer = OPTIMIZER_DICT[self.args.optimizer](meta_optimizer.parameters(), lr=new_lr)
            self.meta_logger.info("Epoch {}: - LEARNING RATE DECAY: changed "
                                  "learning rate from {} to {} <<< ".format(self.epoch,
                                                                            self.learning_rates[-1],
                                                                            new_lr))
            # save new learning rate
            self.learning_rates.append(new_lr)
            if current_loss is not None and current_loss <= 0.8:
                model_path = os.path.join(self.output_dir, meta_optimizer.name + "_eval_run" + str(self.epoch) +
                                          self.config.save_ext)
                meta_optimizer.save_params(model_path)
                self.meta_logger.info("Epoch: {} - Successfully saved model to {}".format(self.epoch, model_path))
        # when lr decay is enabled we make sure that we validate the model each epoch
        if do_lr_decay and self.args.eval_freq != 1:
            self.args.eval_freq = 1
            self.meta_logger.info("Epoch: {} - IMPORTANT: Setting evaluation frequency to 1")

    def init_epoch_stats(self):
        self.epoch_stats["step_losses"][self.epoch] = np.zeros(self.max_time_steps + 1)
        self.epoch_stats["opt_step_hist"][self.epoch] = np.zeros(self.max_time_steps + 1).astype(int)
        self.epoch_stats["halting_step"][self.epoch] = np.zeros(self.max_time_steps + 1).astype(int)
        self.epoch_stats["qt_hist"][self.epoch] = np.zeros(self.max_time_steps)
        # 1) min 2) max 3) mean 4) stddev 5) median
        self.epoch_stats["halting_stats"][self.epoch] = np.zeros(5)
        # only used in mlp experiments, accuracy of MLP on MNIST
        self.epoch_stats["step_acc"][self.epoch] = np.zeros(self.max_time_steps + 1)

    def init_val_stats(self, eval_time_steps=None):
        if eval_time_steps is None:
            eval_time_steps = self.config.max_val_opt_steps
        self.val_stats["step_losses"][self.epoch] = np.zeros(eval_time_steps + 1)
        self.val_stats["step_loss_var"][self.epoch] = np.zeros(eval_time_steps + 1)
        self.val_stats["opt_step_hist"][self.epoch] = np.zeros(eval_time_steps + 1).astype(int)
        self.val_stats["halting_step"][self.epoch] = np.zeros(eval_time_steps + 1).astype(int)
        self.val_stats["qt_hist"][self.epoch] = np.zeros(eval_time_steps)
        # only used in mlp experiments
        self.val_stats["step_acc"][self.epoch] = np.zeros(eval_time_steps + 1)

    def reset_val_stats(self):
        self.val_stats = {"loss": [], "param_error": [], "act_loss": [], "qt_hist": OrderedDict(), "opt_step_hist": {},
                          "step_losses": OrderedDict(), "step_loss_var": OrderedDict(), "opt_loss": [],
                          "step_param_losses": OrderedDict(),
                          "ll_loss": {}, "kl_div": {}, "kl_entropy": {}, "qt_funcs": OrderedDict(),
                          "loss_funcs": [] if (self.args.learner[0:6] != "act_sb" and self.args.learner != "meta_act") else {},
                          "halting_step": OrderedDict(), "kl_term": [], "penalty_term": [], "duration": [],
                          "halt_step_funcs": {}, "mlp_halt_stats": OrderedDict(),
                          "step_acc": OrderedDict()}

    def add_halting_steps(self, halting_steps, is_train=True):
        # expect opt_steps to be an autograd.Variable
        halt_steps_funcs = halting_steps.data.cpu().squeeze().numpy()
        idx_sort = np.argsort(halt_steps_funcs)
        np_array = halt_steps_funcs[idx_sort]
        vals, _, count = np.unique(np_array, return_counts=True, return_index=True)
        if is_train:
            self.epoch_stats["halting_step"][self.epoch][vals.astype(int)] += count.astype(int)
        else:
            self.val_stats["halting_step"][self.epoch][vals.astype(int)] += count.astype(int)
            self.val_stats["halt_step_funcs"][self.epoch] = halt_steps_funcs

    def add_grad_stats(self, mean, stddev):
        self.epoch_stats["grad_stats"][self.epoch-1, 0] = mean
        self.epoch_stats["grad_stats"][self.epoch - 1, 1] = stddev

    def add_opt_steps(self, step, is_train=True):
        # NOTE: we assume step starts with index 0!!! Because of numpy indexing but it is the first time step!!!
        if is_train:
            self.epoch_stats["opt_step_hist"][self.epoch][step] += 1
        else:
            self.val_stats["opt_step_hist"][self.epoch][step] += 1

    def add_step_loss(self, step_loss, step, is_train=True):
        # NOTE: we assume step starts with index 0!!! Because of numpy indexing but it is the first time step!!!
        if not isinstance(step_loss, (np.float, np.float32, np.float64)):
            raise ValueError("step_loss must be a numpy.float but is type {}".format(type(step_loss)))
        if is_train:
            self.epoch_stats["step_losses"][self.epoch][step] += step_loss
        else:
            self.val_stats["step_losses"][self.epoch][step] += step_loss

    def add_step_loss_variance(self, step_loss, step):
        # NOTE: we assume step starts with index 0!!! Because of numpy indexing but it is the first time step!!!
        # if not isinstance(step_loss, (np.float, np.float32, np.float64)):
        #    raise ValueError("step_loss must be a numpy.float but is type {}".format(type(step_loss)))
        loss_var = torch.std(step_loss, 0).data.cpu().numpy()[0].astype(float)
        self.val_stats["step_loss_var"][self.epoch][step] = loss_var

    def add_step_accuracy(self, step_acc, step, is_train=True):
        # NOTE: we assume step starts with index 0!!! Because of numpy indexing but it is the first time step!!!
        if not isinstance(step_acc, (np.float, np.float32, np.float64)):
            raise ValueError("step_loss must be a numpy.float but is type {}".format(type(step_acc)))
        if is_train:
            self.epoch_stats["step_acc"][self.epoch][step] += step_acc
        else:
            self.val_stats["step_acc"][self.epoch][step] += step_acc

    def add_duration(self, epoch_time, is_train=True):
        if not isinstance(epoch_time, (np.float, np.float32, np.float64)):
            raise ValueError("step_loss must be a numpy.float but is type {}".format(type(epoch_time)))
        if is_train:
            self.epoch_stats["duration"].append(epoch_time)
        else:
            self.val_stats["duration"].append(epoch_time)

    def set_regularizer_term(self, reg_term, penalty_term, weight_regularizer):
        if not isinstance(reg_term, (np.float, np.float32, np.float64)):
            raise ValueError("kl_term must be a numpy.float but is type {}".format(type(reg_term)))
        # index minus 1...because epoch counter starts as 1 and we're dealing with np.ndarray
        self.epoch_stats["kl_term"][self.epoch-1] = reg_term
        self.epoch_stats["weight_regularizer"][self.epoch-1] = weight_regularizer
        if penalty_term is not None:
            self.epoch_stats["penalty_term"][self.epoch - 1] = penalty_term

    def add_step_qts(self, qt_values, step=None, is_train=True):
        # NOTE: we assume step starts with index 0!!! Because of numpy indexing but it is the first time step!!!
        #
        if not isinstance(qt_values, np.ndarray):
            raise ValueError("qt_values must be a numpy.float but is type {}".format(type(qt_values)))

        if qt_values.ndim > 1 and step is None:
            if qt_values.shape[0] > 1:
                # take mean over batch size
                qt_values = np.mean(qt_values, axis=0)
            else:
                qt_values = np.squeeze(qt_values, axis=0)
        # print(np.array_str(qt_values[:30], precision=5))
        if step is None:
            start = 0
            end = qt_values.shape[0]
        else:
            start = step
            end = step + 1
        if is_train:
            self.epoch_stats["qt_hist"][self.epoch][start:end] += qt_values
        else:
            self.val_stats["qt_hist"][self.epoch][start:end] += qt_values

    def generate_cost_annealing(self, until_epoch=0):
        r_min = -5
        r_max = 5
        step_size = (r_max - r_min) / float(until_epoch)
        t = np.arange(r_min, r_max, step_size)
        sigmoid = 1. / (1 + np.exp(-t))
        # use this if you want annealing schedule until 0.5 value of the sigmoid function
        until_epoch = sigmoid.shape[0] // 2
        self.annealing_schedule[0:until_epoch] = sigmoid[0:until_epoch]
        # set the rest of the epochs to 0.5 values
        self.annealing_schedule[until_epoch:] = sigmoid[until_epoch]
        # self.annealing_schedule = np.zeros(self.args.max_epoch)

    def generate_curriculum_regime(self, type='linear'):
        if type == 'linear' and self.args.problem[0:10] == "regression":
            self.inc_learning_schedule = np.zeros(self.args.max_epoch).astype(int)
            # we save 1/3 of the epochs to train for args.optimizer_steps e.g. 50
            last_steps = self.args.max_epoch // 3
            off_set = 0
            step_size = 2
            steps = (self.args.max_epoch - last_steps) // step_size
            max_opt_steps = self.args.optimizer_steps - off_set
            slope = max_opt_steps // steps
            for i, idx in enumerate(np.arange(0, (self.args.max_epoch - last_steps), step_size)):
                self.inc_learning_schedule[idx:idx+step_size] = off_set + (slope * (i + 1))

            self.inc_learning_schedule[self.inc_learning_schedule == 0] = self.args.optimizer_steps

        elif type == 'linear' and self.args.problem == "mlp":
            self.inc_learning_schedule = np.zeros(self.args.max_epoch).astype(int)
            self.inc_learning_schedule[0:12] = np.array([3, 4, 6, 10, 20, 40, 50, 60, 70, 80, 90, 100])
            self.inc_learning_schedule[self.inc_learning_schedule == 0] = self.args.optimizer_steps
        else:
            raise ValueError("Curriculum learning scheme is not supported {}".format(type))

    def scale_step_statistics(self, is_train=True):
        if is_train:
            opt_step_array = self.epoch_stats["opt_step_hist"][self.epoch]
        else:
            opt_step_array = self.val_stats["opt_step_hist"][self.epoch]

        step_loss_factors = np.zeros(opt_step_array.shape[0])
        idx_where = np.where(opt_step_array > 0)
        step_loss_factors[idx_where] = 1. / opt_step_array[idx_where]
        if is_train:
            self.epoch_stats["step_losses"][self.epoch] *= step_loss_factors
            # we can (have to) skip the first factor because that is for step 0 and only used for scaling the loss values
            # AND REMEMBER: the loss value array starts with step 0, but qt values start at step 1
            # weigh the step probabilities according to the number of times they are executed
            self.epoch_stats["qt_hist"][self.epoch] *= step_loss_factors[1:]
            # Actually the "meta" model has no qt-values. But we just instead check whether the whole sum is bigger than 0
            # make sure the qt-values sum to one after scaling
            if np.sum(self.epoch_stats["qt_hist"][self.epoch]) > 0:
                self.epoch_stats["qt_hist"][self.epoch] *= 1./np.sum(self.epoch_stats["qt_hist"][self.epoch])
        else:
            self.val_stats["step_losses"][self.epoch] *= step_loss_factors
            # see explanation above
            self.val_stats["qt_hist"][self.epoch] *= step_loss_factors[1:]

    def start(self, meta_logger=None):
        # Model specific things to initialize

        if self.args.problem == "mlp":
            if self.args.learner == "meta":
                # for the meta model we want the the validation horizon to be equal to the max number of opt steps
                # during training. We are actually interest to explore longer horizons during validation but
                # it destroys the learning curves, so we postpone the longer horizons
                self.args.max_val_opt_steps = self.args.optimizer_steps
            else:
                # for the ACT models we don't want the validation horizon to exceed the max horizon during training
                self.args.max_val_opt_steps = self.config.T

        if self.args.learner == "act":
            if self.args.version[0:2] not in ['V1', 'V2']:
                raise ValueError("Version {} currently not supported (only V1.x and V2.x)".format(self.args.version))
            if self.args.fixed_horizon:
                self.avg_num_opt_steps = self.args.optimizer_steps
            else:
                self.avg_num_opt_steps = self.pt_dist.mean
                self.max_time_steps = self.config.T
            self.batch_handler_class = "BatchHandler"
        else:
            if self.args.learner == "act_sb":
                if self.args.version == "V3.2":
                   self.batch_handler_class = "MPACTBatchHandler"
                else:
                    self.batch_handler_class = "ACTBatchHandler"
            elif self.args.learner == "meta_act":
                self.batch_handler_class = "MACTBatchHandler"
            else:
                self.batch_handler_class = "BatchHandler"

            self.avg_num_opt_steps = self.args.optimizer_steps
            if self.args.learner[0:6] == "act_sb" or self.args.learner == "meta_act":
                self.max_time_steps = self.config.T
            if self.args.learner == 'meta' and self.args.problem == "mlp":
                self.validation_handler_class = "ValidateMLPOnMetaLearner"

            if self.args.learner == 'meta' and self.args.version[0:2] == 'V2':
                # Note, we choose here an absolute limit of the horizon, set in the config-file
                self.max_time_steps = self.config.T
                self.avg_num_opt_steps = self.pt_dist.mean
            elif self.args.learner == 'meta' and (self.args.version[0:2] == 'V5' or self.args.version[0:2] == 'V6'):
                # disable BPTT by setting truncated bptt steps to optimizer steps
                self.args.truncated_bptt_step = self.args.optimizer_steps

        # Problem specific initialization things
        if self.args.problem == "rosenbrock":
            assert self.args.learner == "meta", "Rosenbrock problem is only suited for MetaLearner"
            self.config.max_val_opt_steps = self.args.optimizer_steps

        # unfortunately need self.avg_num_opt_steps before we can make the path
        self._set_pathes()
        if meta_logger is None:
            self.meta_logger = create_logger(self, file_handler=True)
        else:
            self.meta_logger = meta_logger
        # if applicable, generate KL cost annealing schedule
        if self.args.learner[0:6] == "act_sb" and self.args.version == "V2":
            self.generate_cost_annealing(int(self.args.max_epoch * self.config.kl_anneal_perc))

        self.fixed_weights = generate_fixed_weights(self)

        # in case we need to evaluate the model, get the test data
        if self.args.eval_freq != 0:
            self.meta_logger.info("Initializing experiment - may take a while to load validation set")
            val_funcs = load_val_data(num_of_funcs=self.config.num_val_funcs, n_samples=self.args.x_samples,
                                      stddev=self.config.stddev, dim=self.args.x_dim, logger=self.meta_logger,
                                      exper=self)
        else:
            val_funcs = None

        # construct the name of the model. Will be used to save model to disk
        if self.args.model == "default":
            if self.args.learner == "meta" or self.args.learner == "act":
                self.args.model = self.args.learner + self.args.version + "_" + self.args.problem + "_" + \
                                   str(int(self.avg_num_opt_steps)) + "ops"
            elif self.args.learner == "meta_act":
                self.args.model = self.args.learner + self.args.version + "_" + self.args.problem + "_" + \
                                  "tau{:.3}".format(self.config.tau)
            else:
                self.args.model = self.args.learner + self.args.version + "_" + self.args.problem + "_" + \
                                   "nu{:.3}".format(self.config.ptT_shape_param)

        self.model_name = self.args.model
        if not self.args.learner == 'manual' and self.args.model is not None:
            self.model_path = os.path.join(self.output_dir, self.args.model + config.save_ext)

        return val_funcs

    def _set_pathes(self):
        if self.args.log_dir == 'default':
            self.args.log_dir = self.config.exper_prefix + \
                                 str.replace(datetime.now(timezone('Europe/Berlin')).strftime(
                                     '%Y-%m-%d %H:%M:%S.%f')[:-7],
                                             ' ', '_') + "_" + create_exper_label(self) + \
                                 "_lr" + "{:.0e}".format(self.args.lr)
            self.args.log_dir = str.replace(str.replace(self.args.log_dir, ':', '_'), '-', '')

        else:
            # custom log dir
            self.args.log_dir = str.replace(self.args.log_dir, ' ', '_')
            self.args.log_dir = str.replace(str.replace(self.args.log_dir, ':', '_'), '-', '')
        log_dir = os.path.join(self.config.log_root_path, self.args.log_dir)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
            fig_path = os.path.join(log_dir, self.config.figure_path)
            os.makedirs(fig_path)
        else:
            # make a back-up copy of the contents
            dst = os.path.join(log_dir, "backup")
            shutil.copytree(log_dir, dst)

        self.output_dir = log_dir

    def eval(self, epoch_obj, meta_optimizer, functions, save_run=None, save_model=True, eval_time_steps=None):
        start_validate = time.time()

        if eval_time_steps is None:
            eval_time_steps = self.config.max_val_opt_steps

        if self.args.problem == "mlp" and self.dta_set is None:
            self.dta_set = MNISTDataSet(self.args.batch_size, use_cuda=self.args.cuda)

        if self.args.problem == "mlp" and self.args.learner == "meta":
            # >>>> Evaluation of MLP with meta model <<<
            self.meta_logger.info("Epoch: {} - Evaluating {} test MLPs".format(self.epoch,
                                                                               len(functions)))
            validation_class = getattr(utils.validation_handler, self.validation_handler_class)
            self.init_val_stats(eval_time_steps)
            validation_handler = validation_class(self, save_model=save_model)
            validation_handler(self, meta_optimizer, functions)
            self.val_stats["step_acc"][self.epoch] = validation_handler.avg_accuracy
            del validation_handler

        elif self.args.learner[0:6] == 'act_sb' or self.args.learner == "meta_act":

            batch_handler_class = getattr(utils.batch_handler, self.batch_handler_class)
            if eval_time_steps is None:
                self.init_val_stats()
            else:
                self.init_val_stats(eval_time_steps)
            if self.args.problem == "mlp":
                num_iters = len(functions)
                num_of_funcs = num_iters
                # initialize a numpy array to store 1) halt step 2) type of mlp (1/2) 3) last loss value
                self.val_stats["mlp_halt_stats"][self.epoch] = np.zeros((num_of_funcs, 3))
                self.val_stats["step_loss_var"][self.epoch] = np.zeros((num_of_funcs, eval_time_steps + 1))
            else:
                num_iters = 1
                num_of_funcs = functions.num_of_funcs

            self.meta_logger.info("Epoch: {} - Evaluating {} test functions".format(self.epoch,
                                                                                    num_of_funcs))
            eval_loss = 0
            kl_term = 0
            penalty_term = 0
            test_max_time_steps_taken = 0
            for i in np.arange(num_iters):
                if num_iters != 0 and i % 10 == 0:
                    print(" >>> Optimizing {} MLP <<<".format(i+1))
                if self.args.problem == "mlp":
                    optimizee = functions[i]
                    if self.args.cuda:
                        optimizee = optimizee.cuda()
                else:
                    optimizee = functions

                optimizee.reset()
                test_batch = batch_handler_class(self, is_train=False, optimizees=optimizee)
                test_batch(self, epoch_obj, meta_optimizer,
                           final_batch=True if self.args.problem == "mlp" else False)
                test_batch.compute_batch_loss(weight_regularizer=epoch_obj.weight_regularizer)
                meta_optimizer.reset_final_loss()
                meta_optimizer.zero_grad()
                eval_loss += test_batch.loss_sum.data.cpu().squeeze().numpy()[0]
                kl_term += test_batch.kl_term
                penalty_term += test_batch.penalty_term
                if epoch_obj.test_max_time_steps_taken > test_max_time_steps_taken:
                    test_max_time_steps_taken = epoch_obj.test_max_time_steps_taken
                if self.args.problem == "mlp":
                    mlp_type = 2 if optimizee.two_hidden_layers else 1
                    h_step = int(self.val_stats["halt_step_funcs"][self.epoch][0])
                    # NOTE: batch_step_losses = max optimization steps but note index 0 = step 1
                    # therefore we substract 1 from halting step
                    last_loss = test_batch.batch_step_losses[h_step - 1].data.cpu().squeeze().numpy()[0]
                    mlp_halt_stats = np.array([h_step, mlp_type, last_loss])
                    # print("Halt stats {}".format(np.array_str(mlp_halt_stats)))
                    # i ==>> index for optimizee
                    self.val_stats["mlp_halt_stats"][self.epoch][i] = mlp_halt_stats
                    self.val_stats["step_loss_var"][self.epoch][i] = test_batch.np_step_losses

            epoch_obj.test_max_time_steps_taken = test_max_time_steps_taken
            eval_loss *= 1/float(num_iters)
            kl_term *= 1 / float(num_iters)
            penalty_term *= 1 / float(num_iters)
            duration = time.time() - start_validate
            self.scale_step_statistics(is_train=False)
            if self.config.max_val_opt_steps > 200:
                start = self.config.max_val_opt_steps - 100
                end = self.config.max_val_opt_steps + 1
                self.meta_logger.info("Epoch: {} - evaluation - showing only last {} time steps".format(self.epoch, 100))
            else:
                start = 0
                end = self.config.max_val_opt_steps + 1
            e_losses = self.val_stats["step_losses"][self.epoch][start:end]
            self.meta_logger.info("Epoch: {} - evaluation result - time step losses".format(self.epoch))
            self.meta_logger.info(np.array_str(e_losses, precision=3))
            # --------------- halting step for this evaluation run --------
            avg_opt_steps, stddev, median, total_steps = self.get_step_dist_statistics(epoch=self.epoch)
            self.meta_logger.info("! - Validation last step {} - !".format(epoch_obj.test_max_time_steps_taken))
            self.meta_logger.info("Epoch: {} - evaluation - halting step distribution".format(self.epoch))
            self.meta_logger.info(np.array_str(self.val_stats["halting_step"][self.epoch]
                                               [1:epoch_obj.test_max_time_steps_taken + 1]))
            self.meta_logger.info("Epoch: {} - evaluation - Average number of optimization steps {:.3f} "
                                  "stddev {:.3f} median {} sum-steps {}".format(self.epoch, avg_opt_steps,
                                                                                stddev, median,
                                                                                int(total_steps)))

            self.meta_logger.info("Epoch: {} - End test evaluation (elapsed time {:.2f} sec) avg act loss/kl-term/penalty "
                                  "{:.3f}/{:.4f}/{:.4f}".format(self.epoch, duration, eval_loss, kl_term,
                                                                penalty_term))
            if self.args.problem == "mlp":
                avg_accuracy = np.mean(test_batch.test_result_scores)
                self.meta_logger.info("Epoch: {} - End test evaluation - Avg accuracy {:.3f}".format(self.epoch,
                                                                                                     avg_accuracy))
                mlp_losses = np.vstack(np.array(mlp.losses) for mlp in functions)
                self.val_stats["loss_funcs"][self.epoch] = mlp_losses
                self.val_stats["step_acc"][self.epoch] = avg_accuracy
                # compute stddev and store results
                np_step_variance = np.std(self.val_stats["step_loss_var"][self.epoch], axis=0)
                self.val_stats["step_loss_var"][self.epoch] = np.zeros(np_step_variance.shape[0])
                self.val_stats["step_loss_var"][self.epoch] = np_step_variance
                # print(np.array_str(self.val_stats["step_loss_var"][self.epoch], precision=3))

            # NOTE: we don't need to scale the step losses because during evaluation run each step is only executed
            # once, which is different during training because we process multiple batches in ONE EPOCH
            # for the validation "loss" (not opt_loss) we can just sum the step_losses we collected earlier
            self.val_stats["loss"].append(np.sum(self.val_stats["step_losses"][self.epoch]))
            # this is the ACT-SB loss!
            self.val_stats["opt_loss"].append(eval_loss)
            # ja not consistent but .kl_term is already a numpy float whereas .loss_sum is not
            self.val_stats["kl_term"].append(kl_term)
            self.val_stats["penalty_term"].append(penalty_term)
            self.add_duration(duration, is_train=False)
            # save the initial distance of each optimizee from its global minimum
            if hasattr(optimizee, "distance_to_min"):
                self.val_stats["loss_funcs"][self.epoch] = optimizee.distance_to_min
            del test_batch

        else:
            # all other models...former ones, still running without Batch object
            if self.args.learner == 'manual':
                # the manual (e.g. SGD, Adam will be validated using full number of optimization steps
                opt_steps = self.args.optimizer_steps
            else:
                opt_steps = self.config.max_val_opt_steps

            validate_optimizer(meta_optimizer, self, val_set=functions,
                               verbose=False,
                               plot_func=False,
                               max_steps=opt_steps,
                               num_of_plots=self.config.num_val_plots,
                               save_qt_prob_funcs=True if self.epoch == self.args.max_epoch else False,
                               save_model=True)
        # can be used for saving intermediate results...before they get lost later...
        if save_run is not None:
            self.save(file_name="exp_stats_run_" + save_run + ".dll")

        if save_model and self.args.learner[0:6] == 'act_sb' or self.args.learner == "meta_act":
            model_path = os.path.join(self.output_dir, meta_optimizer.name + "_eval_run" + str(self.epoch) +
                                      self.config.save_ext)
            meta_optimizer.save_params(model_path)
            self.meta_logger.info("Epoch: {} - Successfully saved model to {}".format(self.epoch, model_path))

    def save(self, file_name=None):
        if file_name is None:
            file_name = "exp_statistics" + ".dll"

        outfile = os.path.join(self.output_dir, file_name)
        logger = self.meta_logger
        optimizer = self.optimizer
        data_set = self.dta_set
        # we set meta_logger temporary to None, because encountering problems when loading the experiment later
        # from file, if the experiment ran on a different machine "can't find .../run.log bla bla
        self.meta_logger = None
        self.optimizer = None
        self.dta_set = None
        with open(outfile, 'wb') as f:
            dill.dump(self, f)
        self.meta_logger = logger
        self.optimizer = optimizer
        self.dta_set = data_set
        if self.meta_logger is not None:
            self.meta_logger.info("Epoch: {} - Saving experimental details to {}".format(self.epoch, outfile))

    def end(self, model):

        if self.model_path is not None:
            if "save_params" in dir(model):
                model.save_params(self.model_path)
            else:
                torch.save(model.state_dict(), self.model_path)

            self.meta_logger.info("Epoch: {} - Successfully saved model parameters to {}".format(self.epoch,
                                                                                                 self.model_path))

        self.save()
        if not self.args.on_server:
            self.generate_figures()

    def generate_figures(self, separate_losses=True):
        if separate_losses:
            # for the MLP experiment we don't want the learning curves of training & validation in one figure
            # because the scales are very different
            loss_plot(self, loss_type="loss", save=True, validation=False)
            loss_plot(self, loss_type="opt_loss", save=True, validation=False, log_scale=False)
            if self.run_validation:
                loss_plot(self, loss_type="loss", save=True, validation=True, only_val=True)
                loss_plot(self, loss_type="opt_loss", save=True, validation=True, only_val=True, log_scale=False)
        else:
            loss_plot(self, loss_type="loss", save=True, validation=self.run_validation)
            loss_plot(self, loss_type="opt_loss", save=True, validation=self.run_validation, log_scale=False)
        plot_image_map_losses(self, data_set="train", do_save=True)
        plot_dist_optimization_steps(self, data_set="train", save=True)
        plot_gradient_stats(self, do_show=False, do_save=True)
        if self.run_validation and self.val_stats["step_losses"].values() != []:
            plot_image_map_losses(self, data_set="eval", do_save=True)

        if self.args.learner[0:6] == "act_sb" or self.args.learner == "meta_act":
            plot_image_map_data(self, data_set="train", data="qt_value", do_save=True, do_show=False)
            plot_image_map_data(self, data_set="train", data="halting_step", do_save=True, do_show=False)
            plot_actsb_qts(self, data_set="train", save=True)
            plot_halting_step_stats_with_loss(self, do_show=False, do_save=True, add_info=True)

            if self.run_validation and self.val_stats["halting_step"].keys() != []:
                if hasattr(self.val_stats, 'halt_step_funcs'):
                    plot_loss_versus_halting_step(self, do_show=False, do_save=True)
                plot_dist_optimization_steps(self, data_set="eval", save=True)
                plot_actsb_qts(self, data_set="eval", save=True)
                plot_image_map_data(self, data_set="eval", data="qt_value", do_save=True, do_show=False)
                plot_image_map_data(self, data_set="eval", data="halting_step", do_save=True, do_show=False)
        if self.args.learner == "act":
            # plot histogram of T distribution (number of optimization steps during training)
            # plot_dist_optimization_steps(self.exper, data_set="train", save=True)
            # plot_dist_optimization_steps(experiment, data_set="val", save=True)
            plot_qt_probs(self, data_set="train", save=True)
            if self.args.problem != "mlp":
                plot_loss_versus_halting_step(self, do_show=False, do_save=True)
            # plot_qt_probs(experiment, data_set="val", save=True, plot_prior=True, height=8, width=8)

    def get_step_dist_statistics(self, epoch=None, with_range=False):
        if epoch is None:
            epoch = self.val_stats["halting_step"].keys()[-1]
        avg_opt_steps, stddev, median, total_steps = halting_step_stats(self.val_stats["halting_step"][epoch])
        if with_range:
            step_indices = np.nonzero(self.val_stats["halting_step"][epoch])
            min_steps = np.min(step_indices)
            max_steps = np.max(step_indices)
            return avg_opt_steps, stddev, median, total_steps, [min_steps, max_steps]

        return avg_opt_steps, stddev, median, total_steps

    @staticmethod
    def load(path_to_exp, full_path=False, do_log=False, meta_logger=None):

        if not full_path:
            path_to_exp = os.path.join(config.log_root_path, os.path.join(path_to_exp, config.exp_file_name))
        else:
            path_to_exp = os.path.join(config.log_root_path, path_to_exp)

        try:
            with open(path_to_exp, 'rb') as f:
                experiment = dill.load(f)

        except IOError as e:
            print "I/O error({0}): {1}".format(e.errno, e.strerror)
            print("Can't open file {}".format(path_to_exp))
            raise IOError
        except:
            print "Unexpected error:", sys.exc_info()[0]
            raise
        # needed to add this for backward compatibility, because added these parameter later
        if not hasattr(experiment.config, 'pT_shape_param'):
            new_config = MetaConfig()
            new_config.__dict__ = experiment.config.__dict__.copy()
            new_config.pT_shape_param = new_config.continue_prob
            new_config.ptT_shape_param = new_config.continue_prob
            experiment.config = new_config
        # we save experiments without their associated logger (because that's seeking trouble when loading the
        # experiment from file. Therefore we need to create a logger in case we need it.
        if experiment.meta_logger is None and do_log and meta_logger:
            if meta_logger is None:
                experiment.meta_logger = create_logger(experiment)
            else:
                experiment.meta_logger = meta_logger
            experiment.meta_logger.info("created local logger for experiment with model {}".format(experiment.model_name))

        # backward compatibility
        if experiment.args.learner == "act_graves":
            experiment.args.learner = "meta_act"

        return experiment
