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
from plots import plot_image_map_data, plot_qt_probs, loss_plot, plot_dist_optimization_steps, plot_gradient_stats
from plots import plot_actsb_qts, plot_image_map_losses, plot_halting_step_stats_with_loss, plot_loss_versus_halting_step
from plots import create_exper_label
import utils.batch_handler
from val_optimizer import validate_optimizer


class Experiment(object):

    def __init__(self, run_args, config):
        self.args = run_args
        self.epoch_stats = {"loss": [], "param_error": [], "act_loss": [], "qt_hist": {}, "opt_step_hist": {},
                            "opt_loss": [], "step_losses": OrderedDict(), "halting_step": {}, "halting_stats": {},
                            "kl_term": np.zeros(run_args.max_epoch),
                            "weight_regularizer": np.zeros(run_args.max_epoch),
                            "grad_stats": np.zeros((run_args.max_epoch, 2)),  # store mean and stddev of gradients model
                            "duration": []}
        self.val_stats = {"loss": [], "param_error": [], "act_loss": [], "qt_hist": {}, "opt_step_hist": {},
                          "step_losses": OrderedDict(), "opt_loss": [],
                          "step_param_losses": OrderedDict(),
                          "ll_loss": {}, "kl_div": {}, "kl_entropy": {}, "qt_funcs": OrderedDict(),
                          "loss_funcs": [] if (run_args.learner[0:6] != "act_sb" and run_args.learner != "act_graves") else {},
                          "halting_step": {}, "kl_term": [], "duration": [], "halt_step_funcs": {}}
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
        if run_args.learner == "act_graves":
            self.annealing_schedule = np.empty(self.args.max_epoch)
            self.annealing_schedule.fill(self.config.tau)
        else:
            self.annealing_schedule = np.ones(self.args.max_epoch)
        if run_args.learner == "meta" and run_args.version == "V7":
            # metaV7 uses incremental learning schedule
            self.generate_incremental_lr_scheme()
        else:
            self.inc_learning_schedule = None
        self.batch_handler_class = None
        self.optimizer = None

    def init_epoch_stats(self):
        self.epoch_stats["step_losses"][self.epoch] = np.zeros(self.max_time_steps + 1)
        self.epoch_stats["opt_step_hist"][self.epoch] = np.zeros(self.max_time_steps + 1).astype(int)
        self.epoch_stats["halting_step"][self.epoch] = np.zeros(self.max_time_steps + 1).astype(int)
        self.epoch_stats["qt_hist"][self.epoch] = np.zeros(self.max_time_steps)
        # 1) min 2) max 3) mean 4) stddev 5) median
        self.epoch_stats["halting_stats"][self.epoch] = np.zeros(5)

    def init_val_stats(self, eval_time_steps=None):
        if eval_time_steps is None:
            eval_time_steps = self.config.max_val_opt_steps
        self.val_stats["step_losses"][self.epoch] = np.zeros(eval_time_steps + 1)
        self.val_stats["opt_step_hist"][self.epoch] = np.zeros(eval_time_steps + 1).astype(int)
        self.val_stats["halting_step"][self.epoch] = np.zeros(eval_time_steps + 1).astype(int)
        self.val_stats["qt_hist"][self.epoch] = np.zeros(eval_time_steps)

    def reset_val_stats(self):
        self.val_stats = {"loss": [], "param_error": [], "act_loss": [], "qt_hist": {}, "opt_step_hist": {},
                          "step_losses": OrderedDict(), "opt_loss": [],
                          "step_param_losses": OrderedDict(),
                          "ll_loss": {}, "kl_div": {}, "kl_entropy": {}, "qt_funcs": OrderedDict(),
                          "loss_funcs": [] if (self.args.learner[0:6] != "act_sb" and self.args.learner != "act_graves") else {},
                          "halting_step": {}, "kl_term": [], "duration": [], "halt_step_funcs": {}}

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

    def add_duration(self, epoch_time, is_train=True):
        if not isinstance(epoch_time, (np.float, np.float32, np.float64)):
            raise ValueError("step_loss must be a numpy.float but is type {}".format(type(epoch_time)))
        if is_train:
            self.epoch_stats["duration"].append(epoch_time)
        else:
            self.val_stats["duration"].append(epoch_time)

    def set_regularizer_term(self, reg_term, weight_regularizer):
        if not isinstance(reg_term, (np.float, np.float32, np.float64)):
            raise ValueError("kl_term must be a numpy.float but is type {}".format(type(reg_term)))
        # index minus 1...because epoch counter starts as 1 and we're dealing with np.ndarray
        self.epoch_stats["kl_term"][self.epoch-1] = reg_term
        self.epoch_stats["weight_regularizer"][self.epoch-1] = weight_regularizer

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
        self.meta_logger.info(">>> Annealing schedule {}".format(np.array_str(self.annealing_schedule)))
        self.meta_logger.info(">>> NOTE: Generated KL cost annealing schedule for first {} epochs <<<".format(until_epoch))

    def generate_incremental_lr_scheme(self, type='linear'):
        if type == 'linear':
            self.inc_learning_schedule = np.zeros(self.args.max_epoch).astype(int)
            off_set = 0  # we always start with 10 steps
            step_size = 2
            steps = self.args.max_epoch // step_size
            max_opt_steps = self.args.optimizer_steps - off_set
            slope = max_opt_steps // steps
            for i, idx in enumerate(np.arange(0, self.args.max_epoch, step_size)):
                self.inc_learning_schedule[idx:idx+step_size] = off_set + (slope * (i + 1))

            self.inc_learning_schedule[self.inc_learning_schedule == 0] = self.args.optimizer_steps

        else:
            raise ValueError("Incremental learning scheme is not supported {}".format(type))

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
            self.epoch_stats["qt_hist"][self.epoch] *= step_loss_factors[1:]
        else:
            self.val_stats["step_losses"][self.epoch] *= step_loss_factors
            # see explanation above
            self.val_stats["qt_hist"][self.epoch] *= step_loss_factors[1:]

    def start(self, meta_logger=None):
        # Model specific things to initialize

        if self.args.learner == "act":
            if self.args.version[0:2] not in ['V1', 'V2']:
                raise ValueError("Version {} currently not supported (only V1.x and V2.x)".format(self.args.version))
            if self.args.fixed_horizon:
                self.avg_num_opt_steps = self.args.optimizer_steps
            else:
                self.avg_num_opt_steps = self.pt_dist.mean
                self.max_time_steps = self.config.T
        else:
            if self.args.learner == "act_sb":
                self.batch_handler_class = "ACTBatchHandler"
            elif self.args.learner == "act_sb_eff":
                self.batch_handler_class = "ACTEfficientBatchHandler"
            elif self.args.learner == "act_graves":
                self.batch_handler_class = "ACTGravesBatchHandler"
            else:
                self.batch_handler_class = "BatchHandler"

            self.avg_num_opt_steps = self.args.optimizer_steps
            if self.args.learner[0:6] == "act_sb" or self.args.learner == "act_graves":
                self.max_time_steps = self.config.T
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
            elif self.args.learner == "act_graves":
                self.args.model = self.args.learner + self.args.version + "_" + self.args.problem + "_" + \
                                  "tau{:.3}".format(self.config.tau)
            else:
                self.args.model = self.args.learner + self.args.version + "_" + self.args.problem + "_" + \
                                   "nu{:.3}".format(self.config.ptT_shape_param)
                self.meta_logger.info("NOTE: >>> Using batch handler class {} <<<".format(self.batch_handler_class))
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
        if self.args.learner[0:6] == 'act_sb' or self.args.learner == "act_graves":
            self.meta_logger.info("Epoch: {} - Evaluating {} test functions".format(self.epoch,
                                                                                    functions.num_of_funcs))
            batch_handler_class = getattr(utils.batch_handler, self.batch_handler_class)
            if eval_time_steps is None:
                self.init_val_stats()
            else:
                self.init_val_stats(eval_time_steps)
            functions.reset()
            test_batch = batch_handler_class(self, is_train=False, optimizees=functions)
            test_batch(self, epoch_obj, meta_optimizer)
            test_batch.compute_batch_loss()
            meta_optimizer.reset_final_loss()
            meta_optimizer.zero_grad()
            eval_loss = test_batch.loss_sum.data.cpu().squeeze().numpy()[0]
            duration = time.time() - start_validate
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
            np_halting_step = self.val_stats["halting_step"][self.epoch]
            avg_opt_steps, stddev, median, total_steps = halting_step_stats(np_halting_step)
            self.meta_logger.info("Epoch: {}, evaluation - halting step distribution".format(self.epoch))
            self.meta_logger.info(np.array_str(np_halting_step[1:epoch_obj.test_max_time_steps_taken + 1]))
            self.meta_logger.info("Epoch: {}, evaluation - Average number of optimization steps {:.3f} "
                                  "stddev {:.3f} median {} sum-steps {}".format(self.epoch, avg_opt_steps,
                                                                                stddev, median,
                                                                                int(total_steps)))

            self.meta_logger.info("Epoch: {} - End test evaluation (elapsed time {:.2f} sec) avg act loss/kl "
                                  "{:.3f}/{:.4f}".format(self.epoch, duration, eval_loss, test_batch.kl_term))
            # NOTE: we don't need to scale the step losses because during evaluation run each step is only executed
            # once, which is different during training because we process multiple batches in ONE EPOCH
            # for the validation "loss" (not opt_loss) we can just sum the step_losses we collected earlier
            self.val_stats["loss"].append(np.sum(self.val_stats["step_losses"][self.epoch]))
            # this is the ACT-SB loss!
            self.val_stats["opt_loss"].append(eval_loss)
            # ja not consistent but .kl_term is already a numpy float whereas .loss_sum is not
            self.val_stats["kl_term"].append(test_batch.kl_term)
            self.add_duration(duration, is_train=False)
            # save the initial distance of each optimizee from its global minimum
            self.val_stats["loss_funcs"][self.epoch] = functions.distance_to_min
            del test_batch

        else:
            # all other models...former ones, still running without Batch object
            if self.args.learner == 'manual':
                # the manual (e.g. SGD, Adam will be validated using full number of optimization steps
                opt_steps = self.args.optimizer_steps
            else:
                opt_steps = self.config.max_val_opt_steps

            validate_optimizer(meta_optimizer, self, val_set=functions, meta_logger=self.meta_logger,
                               verbose=False,
                               plot_func=False,
                               max_steps=opt_steps,
                               num_of_plots=self.config.num_val_plots,
                               save_qt_prob_funcs=True if self.epoch == self.args.max_epoch else False,
                               save_model=True)
        # can be used for saving intermediate results...before they get lost later...
        if save_run is not None:
            self.save(file_name="exp_stats_run_" + save_run + ".dll")

        if save_model and self.args.learner[0:6] == 'act_sb' or self.args.learner == "act_graves":
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
        # we set meta_logger temporary to None, because encountering problems when loading the experiment later
        # from file, if the experiment ran on a different machine "can't find .../run.log bla bla
        self.meta_logger = None
        self.optimizer = None
        with open(outfile, 'wb') as f:
            dill.dump(self, f)
        self.meta_logger = logger
        self.optimizer = optimizer
        if self.meta_logger is not None:
            self.meta_logger.info("Epoch {} - Saving experimental details to {}".format(self.epoch, outfile))

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

    def generate_figures(self):
        loss_plot(self, loss_type="loss", save=True, validation=self.run_validation)  # self.run_validation
        loss_plot(self, loss_type="opt_loss", save=True, validation=self.run_validation, log_scale=False)
        plot_image_map_losses(self, data_set="train", do_save=True)
        plot_dist_optimization_steps(self, data_set="train", save=True)
        plot_gradient_stats(self, do_show=False, do_save=True)
        if self.run_validation:
            plot_image_map_losses(self, data_set="eval", do_save=True)

        if self.args.learner[0:6] == "act_sb" or self.args.learner == "act_graves":
            plot_image_map_data(self, data_set="train", data="qt_value", do_save=True, do_show=False)
            plot_image_map_data(self, data_set="train", data="halting_step", do_save=True, do_show=False)
            plot_actsb_qts(self, data_set="train", save=True)
            plot_halting_step_stats_with_loss(self, do_show=False, do_save=True, add_info=True)

            if self.run_validation:
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
            plot_loss_versus_halting_step(self, do_show=False, do_save=True)
            # plot_qt_probs(experiment, data_set="val", save=True, plot_prior=True, height=8, width=8)

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

        return experiment
