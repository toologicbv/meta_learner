import torch
import numpy as np
import time

from common import construct_prior_p_t_T


def halting_step_stats(halting_steps):
    num_of_steps = halting_steps.shape[0]
    num_of_funcs = np.sum(halting_steps)
    values = np.arange(0, num_of_steps)
    total_steps = np.sum(values * halting_steps)
    avg_opt_steps = int(np.sum(1. / num_of_funcs * values * halting_steps))
    E_x_2 = np.sum(1. / num_of_funcs * values ** 2 * halting_steps)
    stddev = np.sqrt(E_x_2 - avg_opt_steps ** 2)
    cum_sum = np.cumsum(halting_steps)
    if cum_sum[np.nonzero(cum_sum)[0][0]] > num_of_funcs / 2.:
        median = np.nonzero(cum_sum)[0][0]
    else:
        median = np.argmax(cum_sum[cum_sum < num_of_funcs / 2.]) + 1

    return avg_opt_steps, stddev, median, total_steps


class Epoch(object):

    epoch_id = 0

    def __init__(self, exper):
        # want to start at 1
        self.epoch_id = 0
        self.start_time = time.time()
        self.loss_last_time_step = 0.0
        self.final_act_loss = 0.
        self.param_loss = 0.
        self.total_loss_steps = 0.
        self.loss_optimizer = 0.
        self.kl_term = 0.
        self.diff_min = 0.
        self.duration = 0.
        self.avg_opt_steps = []
        self.num_of_batches = exper.args.functions_per_epoch // exper.args.batch_size
        self.prior_probs = construct_prior_p_t_T(exper.args.optimizer_steps, exper.config.ptT_shape_param,
                                                 exper.args.batch_size, exper.args.cuda)
        self.backward_ones = torch.ones(exper.args.batch_size)
        self.train_max_time_steps_taken = 0
        self.test_max_time_steps_taken = 0
        if exper.args.cuda:
            self.backward_ones = self.backward_ones.cuda()

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

    def add_kl_term(self, kl_term):
        if not isinstance(kl_term, (np.float, np.float32, np.float64)):
            raise ValueError("loss must be a numpy.float but is type {}".format(type(kl_term)))
        self.kl_term += kl_term

    def start(self):
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
        self.avg_opt_steps = []
        # prepare epoch variables

    def end(self, exper):
        self.loss_last_time_step *= 1. / float(self.num_of_batches)
        self.param_loss *= 1. / float(self.num_of_batches)
        self.final_act_loss *= 1. / float(self.num_of_batches)
        self.total_loss_steps *= 1. / float(self.num_of_batches)
        self.loss_optimizer *= 1. / float(self.num_of_batches)
        self.kl_term *= 1. / float(self.num_of_batches)

        self.duration = time.time() - self.start_time

        exper.meta_logger.info("Epoch: {}, elapsed time {:.2f} seconds: avg optimizer loss {:.4f} / "
                               "avg total loss (over time-steps) {:.4f} /"
                               " avg final step loss {:.4f} / final-true_min {:.4f}".format(self.epoch_id ,
                                                                                            self.duration,
                                                                                            self.loss_optimizer,
                                                                                            self.total_loss_steps,
                                                                                            self.loss_last_time_step,
                                                                                            self.diff_min))
        if exper.args.learner == 'act':
            exper.meta_logger.info("Epoch: {}, ACT - average final act_loss {:.4f}".format(self.epoch_id,
                                                                                           self.final_act_loss))
            avg_opt_steps = int(np.mean(np.array(self.avg_opt_steps)))
            exper.meta_logger.debug("Epoch: {}, Average number of optimization steps {}".format(self.epoch_id + 1,
                                                                                                avg_opt_steps))
        if exper.args.learner == 'meta' and exper.args.version[0:2] == "V2":
            avg_opt_steps = int(np.mean(np.array(self.avg_opt_steps)))
            exper.meta_logger.info("Epoch: {}, Average number of optimization steps {}".format(self.epoch_id,
                                                                                               avg_opt_steps))
        if exper.args.learner == 'act_sb':
            np_array = exper.epoch_stats["halting_step"][self.epoch_id]
            avg_opt_steps, stddev, median, total_steps = halting_step_stats(np_array)
            e_losses = exper.epoch_stats["step_losses"][self.epoch_id][0:self.train_max_time_steps_taken+1]
            exper.meta_logger.info("time step losses")
            exper.meta_logger.info(np.array_str(e_losses,  precision=3))
            exper.meta_logger.debug("qt values")
            exper.meta_logger.debug(np.array_str(exper.epoch_stats["qt_hist"][self.epoch_id]
                                                [0:self.train_max_time_steps_taken + 1],  precision=4))
            exper.meta_logger.info("halting step frequencies")
            exper.meta_logger.info(np.array_str(np_array[0:self.train_max_time_steps_taken+1]))

            exper.meta_logger.info("Epoch: {}, Average number of optimization steps {} "
                                   "stddev {:.3f} median {} sum-steps {}".format(self.epoch_id,
                                                                                 avg_opt_steps,
                                                                                 stddev, median,
                                                                                 int(total_steps)))
            exper.meta_logger.info("Epoch: {}, ACT-SB - optimizer-loss/kl-term {:.4f}"
                                   "/{:.4f}".format(self.epoch_id, self.loss_optimizer, self.kl_term))

        exper.epoch_stats["loss"].append(self.total_loss_steps)
        exper.epoch_stats["param_error"].append(self.param_loss)
        exper.add_duration(self.duration, is_train=True)
        if exper.args.learner[0:3] == 'act':
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
