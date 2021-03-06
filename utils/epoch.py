import os

import torch
import numpy as np
import time

from common import construct_prior_p_t_T, halting_step_stats


class Epoch(object):

    epoch_id = 0

    def __init__(self):
        # want to start at 1
        self.epoch_id = 0
        self.start_time = time.time()
        self.loss_last_time_step = 0.0
        self.final_act_loss = 0.
        self.param_loss = 0.
        self.total_loss_steps = 0.
        self.loss_optimizer = 0.
        self.kl_term = 0.
        self.penalty_term = 0.
        self.diff_min = 0.
        self.duration = 0.
        self.avg_opt_steps = []
        self.num_of_batches = 0
        self.prior_probs = None
        self.train_max_time_steps_taken = 0
        self.test_max_time_steps_taken = 0
        self.weight_regularizer = 0
        self.backward_ones = None
        self.model_grads = []
        self.lr_decay_last_epoch = 0

    def add_step_loss(self, avg_loss, last_time_step=False):
        if not isinstance(avg_loss, (np.float, np.float32, np.float64)):
            raise ValueError("avg_loss must be a numpy.float but is type {}".format(type(avg_loss)))
        self.total_loss_steps += avg_loss
        if last_time_step:
            self.loss_last_time_step += avg_loss

    def add_act_loss(self, loss):
        if not isinstance(loss, (np.float, np.float32, np.float64)):
            raise ValueError("loss must be a numpy.float but is type {}".format(type(loss)))
        self.final_act_loss += loss
        self.loss_optimizer += loss

    def add_kl_term(self, kl_term, penalty_term=None):
        if not isinstance(kl_term, (np.float, np.float32, np.float64)):
            raise ValueError("loss must be a numpy.float but is type {}".format(type(kl_term)))
        self.kl_term += kl_term
        if penalty_term is not None:
            self.penalty_term += penalty_term

    def start(self, exper):
        Epoch.epoch_id += 1
        self.epoch_id = Epoch.epoch_id
        self.start_time = time.time()
        self.loss_last_time_step = 0
        self.final_act_loss = 0
        self.param_loss = 0.
        self.total_loss_steps = 0.
        self.loss_optimizer = 0
        self.diff_min = 0.
        self.duration = 0.
        self.num_of_batches = exper.args.functions_per_epoch // exper.args.batch_size
        self.prior_probs = construct_prior_p_t_T(exper.args.optimizer_steps, exper.config.ptT_shape_param,
                                                 exper.args.batch_size, exper.args.cuda)
        self.backward_ones = torch.ones(exper.args.batch_size)
        if exper.args.cuda:
            self.backward_ones = self.backward_ones.cuda()

        self.avg_opt_steps = []
        if exper.annealing_schedule is not None:
            self.weight_regularizer = float(exper.annealing_schedule[self.epoch_id-1])
        # prepare epoch variables

    def end(self, exper):
        self.loss_last_time_step *= 1. / float(self.num_of_batches) * 1./float(exper.args.samples_per_batch)
        self.param_loss *= 1. / float(self.num_of_batches) * 1./float(exper.args.samples_per_batch)
        self.total_loss_steps *= 1. / float(self.num_of_batches) * 1./float(exper.args.samples_per_batch)
        self.loss_optimizer *= 1. / float(self.num_of_batches)
        self.final_act_loss *= 1. / float(self.num_of_batches)
        self.kl_term *= 1. / float(self.num_of_batches)
        self.penalty_term *= 1. / float(self.num_of_batches)
        exper.set_regularizer_term(self.kl_term, self.penalty_term, self.weight_regularizer)

        self.duration = time.time() - self.start_time

        exper.meta_logger.info("Epoch: {}, elapsed time {:.2f} seconds: avg optimizer loss {:.4f} / "
                               "avg total loss (over time-steps) {:.4f} /"
                               " avg final step loss {:.4f} / final-true_min {:.4f}".format(self.epoch_id ,
                                                                                            self.duration,
                                                                                            self.loss_optimizer,
                                                                                            self.total_loss_steps,
                                                                                            self.loss_last_time_step,
                                                                                            self.diff_min))
        model_grads = np.array(self.model_grads)
        model_grads_mean = np.mean(model_grads)
        model_grads_stddev = np.std(model_grads)
        exper.add_grad_stats(model_grads_mean, model_grads_stddev)
        exper.meta_logger.info("Epoch: {}, gradient statistics - mean={:.3f} / stddev={:.3f}".format(self.epoch_id,
                                                                                                     model_grads_mean,
                                                                                                     model_grads_stddev))
        if exper.args.learner == 'act':
            exper.meta_logger.info("Epoch: {}, ACT - average final act_loss {:.4f}".format(self.epoch_id,
                                                                                           self.final_act_loss))
            avg_opt_steps = int(np.mean(np.array(self.avg_opt_steps)))
            exper.meta_logger.debug("Epoch: {}, Average number of optimization steps {}".format(self.epoch_id + 1,
                                                                                                avg_opt_steps))
        if exper.args.learner == "meta" or (exper.args.learner == "act" and exper.args.version == "V2"):
            e_losses = exper.epoch_stats["step_losses"][self.epoch_id][0:exper.max_time_steps + 1]
            exper.meta_logger.info("time step losses")
            exper.meta_logger.info(np.array_str(e_losses, precision=4))

        if exper.args.learner == 'meta' and exper.args.version[0:2] == "V2":
            avg_opt_steps = int(np.mean(np.array(self.avg_opt_steps)))
            exper.meta_logger.info("Epoch: {}, Average number of optimization steps {}".format(self.epoch_id,
                                                                                               avg_opt_steps))
        if exper.args.learner[0:6] == 'act_sb' or exper.args.learner == "meta_act":
            np_halting_step = exper.epoch_stats["halting_step"][self.epoch_id]
            step_indices = np.nonzero(np_halting_step)
            min_steps = np.min(step_indices)
            max_steps = np.max(step_indices)
            avg_opt_steps, stddev, median, total_steps = halting_step_stats(np_halting_step)

            exper.epoch_stats["halting_stats"][self.epoch_id] = np.array([min_steps, max_steps, avg_opt_steps, stddev, median])
            # here we need to add 1 to max_time_steps_taken because the step_losses starts with the step 0,
            # the first value of the optimizees before we start optimizing them
            e_losses = exper.epoch_stats["step_losses"][self.epoch_id][0:self.train_max_time_steps_taken+1]
            exper.meta_logger.info("time step losses")
            exper.meta_logger.info(np.array_str(e_losses,  precision=3))
            exper.meta_logger.debug("qt values")

            exper.meta_logger.debug(np.array_str(exper.epoch_stats["qt_hist"][self.epoch_id]
                                                [0:self.train_max_time_steps_taken],  precision=4))
            exper.meta_logger.info("halting step frequencies - "
                                   "NOTE max steps taken {}".format(self.train_max_time_steps_taken))

            exper.meta_logger.info(np.array_str(np_halting_step[1:self.train_max_time_steps_taken+1]))

            exper.meta_logger.info("Epoch: {}, Average number of optimization steps {:.3f} "
                                   "stddev {:.3f} median {} sum-steps {}".format(self.epoch_id,
                                                                                 avg_opt_steps,
                                                                                 stddev, median,
                                                                                 int(total_steps)))
            exper.meta_logger.info("Epoch: {}, ACT-SB(klw={:.5f}) - optimizer-loss/kl-term/penalty-term {:.4f}"
                                   "/{:.4f}/{:.7f}".format(self.epoch_id, self.weight_regularizer, self.loss_optimizer,
                                                           self.kl_term, self.penalty_term))

        exper.epoch_stats["loss"].append(self.total_loss_steps)
        exper.epoch_stats["param_error"].append(self.param_loss)
        exper.add_duration(self.duration, is_train=True)

        if exper.args.learner[0:3] == 'act' or exper.args.learner == "meta_act":
            exper.epoch_stats["opt_loss"].append(self.final_act_loss)
        elif exper.args.learner == 'meta':
            exper.epoch_stats["opt_loss"].append(self.loss_optimizer)

    def set_max_time_steps_taken(self, steps, is_train):
        if is_train:
            self.train_max_time_steps_taken = steps
        else:
            self.test_max_time_steps_taken = steps

    def get_max_time_steps_taken(self, is_train):
        if is_train:
            return self.train_max_time_steps_taken
        else:
            return self.test_max_time_steps_taken

    def execute_checkpoint(self, exper, meta_optimizer):
        checkpoint_dir = os.path.join(exper.output_dir, exper.args.checkpoint_dir)
        if not os.path.isdir(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        model_path = os.path.join(checkpoint_dir, meta_optimizer.name + "_chkpt" + str(self.epoch_id) +
                                  exper.config.save_ext)
        meta_optimizer.save_params(model_path)
        exper.meta_logger.info("Epoch {}: - checkpoint - successfully saved model to {}".format(self.epoch_id, model_path))
