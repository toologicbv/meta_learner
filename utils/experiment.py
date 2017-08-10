import torch
import os
import dill
import shutil
from collections import OrderedDict
from datetime import datetime
from pytz import timezone
import numpy as np

from config import config
from probs import TimeStepsDist
from common import load_val_data, create_logger, generate_fixed_weights, create_exper_label
from plots import plot_image_training_data, plot_qt_probs, loss_plot, plot_dist_optimization_steps
from plots import plot_actsb_qts, plot_image_training_loss
from batch_handler import ACTBatchHandler
from val_optimizer import validate_optimizer


class Experiment(object):

    def __init__(self, run_args, config=None):
        self.args = run_args
        self.epoch_stats = {"loss": [], "param_error": [], "act_loss": [], "qt_hist": {}, "opt_step_hist": {},
                            "opt_loss": [], "step_losses": OrderedDict(), "halting_step": {},
                            "kl_term": np.zeros(run_args.max_epoch),
                            "kl_weight": np.zeros(run_args.max_epoch)}
        self.val_stats = {"loss": [], "param_error": [], "act_loss": [], "qt_hist": {}, "opt_step_hist": {},
                          "step_losses": OrderedDict(), "opt_loss": [],
                          "step_param_losses": OrderedDict(),
                          "ll_loss": {}, "kl_div": {}, "kl_entropy": {}, "qt_funcs": OrderedDict(),
                          "loss_funcs": [], "halting_step": {}}
        self.epoch = 0
        self.output_dir = None
        self.model_path = None
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

    def init_val_stats(self):
        self.val_stats["step_losses"][self.epoch] = np.zeros(self.config.max_val_opt_steps + 1)
        self.val_stats["opt_step_hist"][self.epoch] = np.zeros(self.config.max_val_opt_steps + 1).astype(int)
        self.val_stats["halting_step"][self.epoch] = np.zeros(self.config.max_val_opt_steps + 1).astype(int)
        self.val_stats["qt_hist"][self.epoch] = np.zeros(self.config.max_val_opt_steps)

    def reset_val_stats(self):
        self.val_stats = {"loss": [], "param_error": [], "act_loss": [], "qt_hist": {}, "opt_step_hist": {},
                          "step_losses": OrderedDict(),
                          "step_param_losses": OrderedDict(), "ll_loss": {}, "kl_div": {}, "kl_entropy": {},
                          "qt_funcs": OrderedDict(), "loss_funcs": [], "opt_loss": []}

    def add_halting_steps(self, halting_steps, is_train=True):
        # expect opt_steps to be an autograd.Variable
        np_array = halting_steps.data.cpu().squeeze().numpy()
        idx_sort = np.argsort(np_array)
        np_array = np_array[idx_sort]
        vals, _, count = np.unique(np_array, return_counts=True, return_index=True)
        if is_train:
            self.epoch_stats["halting_step"][self.epoch][vals.astype(int)] += count.astype(int)
        else:
            self.val_stats["halting_step"][self.epoch][vals.astype(int)] += count.astype(int)

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

    def set_kl_term(self, kl_term, kl_weight):
        if not isinstance(kl_term, (np.float, np.float32, np.float64)):
            raise ValueError("kl_term must be a numpy.float but is type {}".format(type(kl_term)))
        # index minus 1...because epoch counter starts as 1 and we're dealing with np.ndarray
        self.epoch_stats["kl_term"][self.epoch-1] = kl_term
        self.epoch_stats["kl_weight"][self.epoch-1] = kl_weight

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

    def scale_step_losses(self, is_train=True):
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

    def start(self, epoch_obj):

        if self.args.learner == "act":
            if self.args.version[0:2] not in ['V1', 'V2']:
                raise ValueError("Version {} currently not supported (only V1.x and V2.x)".format(self.args.version))

            if self.args.fixed_horizon:
                self.avg_num_opt_steps = self.args.optimizer_steps

            else:
                self.avg_num_opt_steps = self.pt_dist.mean
                self.max_time_steps = self.config.T
        else:
            self.avg_num_opt_steps = self.args.optimizer_steps
            if self.args.learner == 'meta' and self.args.version[0:2] == 'V2':
                # Note, we choose here an absolute limit of the horizon, set in the config-file
                self.max_time_steps = self.config.T
                self.avg_num_opt_steps = self.pt_dist.mean
            elif self.args.learner == 'meta' and (self.args.version[0:2] == 'V5' or self.args.version[0:2] == 'V6'):
                # disable BPTT by setting truncated bptt steps to optimizer steps
                self.args.truncated_bptt_step = self.args.optimizer_steps

        if self.args.problem == "rosenbrock":
            assert self.args.learner == "meta", "Rosenbrock problem is only suited for MetaLearner"
            self.config.max_val_opt_steps = self.args.optimizer_steps
        # unfortunately need self.avg_num_opt_steps before we can make the path
        self._set_pathes()
        self.meta_logger = create_logger(self, file_handler=True)
        self.meta_logger.info("Initializing experiment - may take a while to load validation set")
        self.fixed_weights = generate_fixed_weights(self)

        if self.args.eval_freq != 0:
            val_funcs = load_val_data(num_of_funcs=self.config.num_val_funcs, n_samples=self.args.x_samples,
                                      stddev=self.config.stddev, dim=self.args.x_dim, logger=self.meta_logger,
                                      exper=self)
        else:
            val_funcs = None

        return val_funcs

    def _set_pathes(self):
        if self.args.log_dir == 'default':
            self.args.log_dir = self.config.exper_prefix + \
                                 str.replace(datetime.now(timezone('Europe/Berlin')).strftime(
                                     '%Y-%m-%d %H:%M:%S.%f')[:-7],
                                             ' ', '_') + "_" + create_exper_label(self) + \
                                 "_lr" + "{:.0e}".format(self.args.lr) + "_" + self.args.optimizer
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

    def eval(self, epoch_obj, meta_optimizer, functions):
        if self.args.learner == 'act_sb':

            self.init_val_stats()
            test_batch = ACTBatchHandler(self, is_train=False, optimizees=functions)
            test_batch(self, epoch_obj, meta_optimizer)
            test_batch.compute_batch_loss()
            meta_optimizer.reset_final_loss()
            meta_optimizer.zero_grad()
            eval_loss = test_batch.loss_sum.data.cpu().squeeze().numpy()[0]

            self.meta_logger.info("Epoch {} - End test evaluation avg act loss {:.3f}".format(self.epoch,
                                                                                              eval_loss))
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

    def save(self, file_name=None):
        if file_name is None:
            file_name = "exp_statistics" + ".dll"

        outfile = os.path.join(self.output_dir, file_name)

        with open(outfile, 'wb') as f:
            dill.dump(self, f)
        print("INFO - Successfully saved experimental details to {}".format(outfile))

    def end(self, model):
        if not self.args.learner == 'manual' and model.name is not None:
            model_path = os.path.join(self.output_dir, model.name + config.save_ext)
            torch.save(model.state_dict(), model_path)
            self.model_path = model_path
            print("INFO - Successfully saved model parameters to {}".format(model_path))

        self.save()
        if not self.args.on_server:
            loss_plot(self, loss_type="loss", save=True, validation=self.run_validation)
            loss_plot(self, loss_type="opt_loss", save=True, validation=self.run_validation, log_scale=False)
            plot_image_training_loss(self, do_save=True)
            if self.args.learner == "act_sb":
                plot_image_training_data(self, data="qt_value", do_save=True, do_show=False)
                plot_image_training_data(self, data="halting_step", do_save=True, do_show=False)
                plot_actsb_qts(self, data_set="train", save=True)
                if self.run_validation:
                    plot_dist_optimization_steps(self, data_set="val", save=True)
            if self.args.learner == "act":
                # plot histogram of T distribution (number of optimization steps during training)
                # plot_dist_optimization_steps(self.exper, data_set="train", save=True)
                # plot_dist_optimization_steps(experiment, data_set="val", save=True)
                plot_qt_probs(self, data_set="train", save=True)
                # plot_qt_probs(experiment, data_set="val", save=True, plot_prior=True, height=8, width=8)
                # param_error_plot(experiment, save=True)