import abc
import numpy as np
from scipy.stats import geom, nbinom
import torch
from torch.autograd import Variable


from utils.common import get_batch_functions, get_func_loss
from utils.helper import tensor_and, tensor_any


from models.rnn_optimizer import get_step_loss, kl_divergence


class BatchHandler(object):
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def cuda(self):
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def backward(self, *args):
        pass


class ACTBatchHandler(BatchHandler):

    # static class variable to count the batches
    id = 0

    def __init__(self, exper, is_train, optimizees=None):

        self.is_train = is_train
        self.type_prior = exper.type_prior
        self.prior_shape_param1 = exper.config.ptT_shape_param
        if exper.args.version == "V3": # only for neg-binomial prior
            self.prior_shape_param2 = exper.config.num_of_successes
        if self.is_train:
            self.functions = get_batch_functions(exper)
            self.horizon = exper.config.T
        else:
            self.horizon = exper.config.max_val_opt_steps
            if optimizees is None:
                raise ValueError("Parameter -optimizees- can't be None. Please provide a test set")
            self.functions = optimizees

        self.batch_size = self.functions.num_of_funcs
        # will be intensively used to select the functions that still need to be optimzed after t steps
        self.bool_mask = Variable(torch.ones(self.batch_size, 1).type(torch.ByteTensor))
        self.float_mask = Variable(torch.ones(self.batch_size, 1))
        self.max_T = Variable(torch.FloatTensor([self.horizon-1]).expand_as(self.float_mask))
        if self.is_train:
            self.one_minus_eps = torch.FloatTensor(self.max_T.size()).uniform_(0., 1.)
            self.one_minus_eps = Variable(1. - self.one_minus_eps)
        else:
            # during evaluation we fix the threshold
            self.one_minus_eps = Variable(torch.zeros(self.max_T.size()))
            self.one_minus_eps[:] = exper.config.qt_threshold

        # IMPORTANT, this tensor needs to have the same size as exper.epoch_stats["opt_step_hist"][exper.epoch] which
        # is set in train_optimizer.py in the beginning of the epoch
        self.halting_steps = Variable(torch.zeros(self.max_T.size()))
        self.counter_compare = Variable(torch.zeros(self.max_T.size()))
        #
        self.one_minus_qt = Variable(torch.zeros(self.batch_size, self.horizon))
        self.qts = Variable(torch.zeros(self.batch_size, self.horizon))
        # IMPORTANT: we're using cum_qt only for comparison in the WHILE loop in order to determine when to stop
        self.compare_probs = Variable(torch.zeros(self.max_T.size()))
        self.remainder_probs = Variable(torch.zeros(self.max_T.size()))
        self.time_step = 0
        self.step = 0
        self.loss_sum = 0
        self.kl_term = 0
        self.batch_step_losses = []
        self.tensor_one = Variable(torch.ones(1))
        if exper.args.learner == "act_sb" and exper.args.learner == "V4":
            self.with_meta_loss = True
        else:
            self.with_meta_loss = False
        self.sum_step_losses = []
        self.backward_ones = torch.ones(self.batch_size)
        # only used during evaluation to capture the last time step when at least one optimizee still needed processing
        self.eval_last_step_taken = 0.
        if exper.args.cuda:
            self.cuda()

    def cuda(self):
        self.bool_mask = self.bool_mask.cuda()
        self.float_mask = self.float_mask.cuda()
        self.max_T = self.max_T.cuda()
        self.one_minus_eps = self.one_minus_eps.cuda()
        self.halting_steps = self.halting_steps.cuda()
        self.counter_compare = self.counter_compare.cuda()
        self.one_minus_qt = self.one_minus_qt.cuda()
        self.qts = self.qts.cuda()
        self.compare_probs = self.compare_probs.cuda()
        self.remainder_probs = self.remainder_probs.cuda()
        self.tensor_one = self.tensor_one.cuda()
        self.backward_ones = self.backward_ones.cuda()

    def act_step(self, exper, meta_optimizer):
        """
        Subtleties of the batch processing. TODO: NEED TO EXPLAIN THIS!!!
        :param exper:
        :param meta_optimizer:
        :return: IMPORTANT RETURNS LOSS STEP THAT IS NOT MULTIPLIED BY QT-VALUE! NUMPY FLOAT32
        """
        loss = get_func_loss(exper, self.functions, average=False)
        # compute gradients of optimizee which will need for the meta-learner
        loss.backward(self.backward_ones)
        # register the baseline loss at step 0
        if self.step == 0:
            self.process_step0(exper, loss)
        param_size = self.functions.params.grad.size()
        if not self.is_train:
            # IMPORTANT! BECAUSE OTHERWISE RUNNING INTO MEMORY ISSUES - PASS GRADS NOT IN A NEW VARIABLE
            delta_param, delta_qt = meta_optimizer.forward(self.functions.params.grad.view(-1))
        else:
            flat_grads = Variable(self.functions.params.grad.data.view(-1))
            delta_param, delta_qt = meta_optimizer.forward(flat_grads)
        # (1) reshape parameter tensor (2) take mean to compute qt values
        delta_param = delta_param.view(param_size)
        delta_qt = torch.mean(delta_qt.view(*param_size), 1, keepdim=True)
        if not self.is_train:
            # during evaluation we keep ALL parameters generated in order to be able to compute new loss values
            # for all optimizees
            eval_par_new = Variable(self.functions.params.data - delta_param.data)
        # then, apply the previous batch mask, although we did the forward pass with all functions, we filter here
        delta_param = torch.mul(delta_param, self.float_mask)
        qt_new = torch.mul(delta_qt, self.float_mask)
        # compute new probability values, based on the cumulative probs construct new batch mask
        new_probs = self.compute_probs(qt_new)
        # we need to determine the indices of the functions that "stop" in this time step (new_funcs_mask)
        # for those functions the last step probability needs to be calculated different
        funcs_that_stop = torch.le(self.compare_probs + new_probs, self.one_minus_eps)
        new_batch_mask = tensor_and(funcs_that_stop, self.bool_mask)
        new_float_mask = new_batch_mask.float()
        # increase the number of steps taken for the functions that are still in the race, we need these to compare
        # against the maximum allowed number of steps
        self.counter_compare += new_float_mask
        # and we increase the number of time steps TAKEN with the previous float_mask object to keep track of the steps
        self.halting_steps += self.float_mask
        # IMPORTANT: although above we constructed the new mask (for the next time step) we use the previous mask for the
        # increase of the cumulative probs which we use in the WHILE loop to determine when to stop the complete batch
        self.compare_probs += torch.mul(new_probs, self.float_mask)
        par_new = self.functions.params - delta_param
        # generate object that holds time_step_condition (which function has taken more than MAX time steps allowed)
        time_step_condition = torch.le(self.counter_compare, self.max_T)
        # we then generate the mask that determines which functions will finally be part of the next round
        next_iteration_condition = tensor_and(new_batch_mask, time_step_condition)
        final_func_mask = (self.bool_mask - next_iteration_condition) == 1
        # set the remaining probs for all functions that participate in the next time step
        # we use this object for determining the rest-probability for the functions that stop after this time step
        # the mask "final_func_mask" holds the indices of the functions that stop after this time step
        # the mask "next_iteration_condition" holds the indices of the functions that continue in the next time step
        self.remainder_probs += torch.mul(new_probs, next_iteration_condition.float())
        self.set_qt_values(new_probs, next_iteration_condition, final_func_mask)
        # set the masks for the next time step
        self.float_mask = new_float_mask
        self.bool_mask = new_float_mask.type(torch.cuda.ByteTensor) if exper.args.cuda else \
            new_float_mask.type(torch.ByteTensor)
        # compute the new step loss
        if self.is_train:
            loss_step = self.step_loss(par_new, average_batch=True)
        else:
            loss_step = self.step_loss(eval_par_new, average_batch=True)
        # update batch functions parameter for next step
        if self.is_train:
            self.functions.set_parameters(par_new)
        else:
            self.functions.set_parameters(eval_par_new)

        self.functions.params.grad.data.zero_()

        return loss_step.data.cpu().squeeze().numpy()[0]

    def __call__(self, exper, epoch_obj, meta_optimizer):

        self.step = 0
        meta_optimizer.reset_lstm(keep_states=False)
        if self.is_train:
            do_continue = tensor_any(tensor_and(torch.le(self.compare_probs, self.one_minus_eps),
                                                torch.le(self.counter_compare, self.max_T))).data.cpu().numpy()[0]
        else:
            do_continue = True if self.step < self.horizon-1 else False

        while do_continue:
            # IMPORTANT! avg_loss_step is NOT MULTIPLIED BY THE qt-value!!!!
            avg_loss_step = self.act_step(exper, meta_optimizer)
            # le1 = np.sum(torch.le(self.cum_qt, self.one_minus_eps).data.cpu().numpy())
            # le2 = np.sum(torch.le(self.counter_compare, self.max_T).data.cpu().numpy())
            if self.is_train:
                do_continue = tensor_any(
                    tensor_and(torch.le(self.compare_probs, self.one_minus_eps),
                               torch.le(self.counter_compare, self.max_T))).data.cpu().numpy()[0]
            else:
                do_continue = True if self.step < self.horizon-1 else False
                if self.eval_last_step_taken == 0:
                    num_next_iter = torch.sum(self.float_mask).data.cpu().squeeze().numpy()[0]
                    if num_next_iter == 0. and self.eval_last_step_taken == 0.:
                        self.eval_last_step_taken = self.step + 1

            # important increase step after "do_continue" stuff but before adding step losses
            self.step += 1
            exper.add_step_loss(avg_loss_step, self.step, is_train=self.is_train)
            exper.add_opt_steps(self.step, is_train=self.is_train)
            epoch_obj.add_step_loss(avg_loss_step, last_time_step=not do_continue)

        exper.add_step_qts(self.qts[:, 0:self.step].data.cpu().numpy(), is_train=self.is_train)
        exper.add_halting_steps(self.halting_steps, is_train=self.is_train)
        # set the class variable if we reached a new maximum time steps
        if epoch_obj.get_max_time_steps_taken(self.is_train) < self.step:
            epoch_obj.set_max_time_steps_taken(self.step, self.is_train)
        if not self.is_train:
            if self.eval_last_step_taken == 0:
                self.eval_last_step_taken = self.step
            epoch_obj.set_max_time_steps_taken(self.eval_last_step_taken, self.is_train)
            exper.meta_logger.info("! - Validation last step {} - !".format(self.eval_last_step_taken))

    def compute_probs(self, qts):

        # if this is the first time we broke the stick, qts are just our probs
        if self.step == 0:
            probs = qts
        else:
            # stick-breaking procedure: the new probs = \prod_{i=1}^{t-1} (1 - qt_i) qt_t
            probs = torch.mul(torch.prod(self.one_minus_qt[:, 0:self.step], 1, keepdim=True), qts)

        self.one_minus_qt[:, self.step] = self.tensor_one - Variable(qts.data.squeeze())
        return probs

    def backward(self, epoch_obj, meta_optimizer, optimizer):
        if len(self.batch_step_losses) == 0:
            raise RuntimeError("No batch losses accumulated. Can't execute backward() on object")
        self.compute_batch_loss(kl_weight=epoch_obj.kl_weight)
        epoch_obj.add_kl_term(self.kl_term)
        self.loss_sum.backward()
        optimizer.step()
        meta_optimizer.reset_final_loss()
        meta_optimizer.zero_grad()
        return self.loss_sum.data.cpu().squeeze().numpy()[0]

    def step_loss(self, new_parameters, average_batch=True):

        # Note: for the ACT step loss, we're only summing over the number of samples (dim1), for META model
        # we also sum over dim0 - the number of functions. But for ACT we need the losses per function in the
        # final_loss calculation (multiplied with the qt values, which we collect also for each function
        loss = get_step_loss(self.functions, new_parameters, avg_batch=False)
        if loss.dim() == 1:
            loss = loss.unsqueeze(1)

        qts = self.qts[:, self.step].unsqueeze(1).double()
        elbo_loss = torch.mean(torch.mul(loss.double(), qts), 0, keepdim=True)
        self.batch_step_losses.append(elbo_loss)
        if self.with_meta_loss:
            self.sum_step_losses.append(torch.mean(loss, 0, keepdim=True))
        if average_batch:
            return torch.mean(loss)
        else:
            return loss

    def compute_batch_loss(self, kl_weight=1.):
        # construct the individual priors for each batch function, return DoubleTensor[batch_size, self.steps]
        g_priors = self.construct_priors()
        # truncate the self.qts variable to the maximum number of opt steps we made
        if self.is_train:
            qts = self.qts[:, 0:self.step].double()
        else:
            qts = self.qts[:, 0:self.eval_last_step_taken].double()
            g_priors = g_priors[:, 0:self.eval_last_step_taken]
        # compute KL divergence term
        kl_term = kl_weight * torch.mean(kl_divergence(qts, g_priors))
        self.kl_term = kl_term.data.cpu().squeeze().numpy()[0]
        # compute final loss, in which we multiply each loss by the qt time step values
        losses = torch.cat(self.batch_step_losses, 1).double()
        # sum over the time steps (dim 1) and average over the batch dimension
        # losses = torch.mean(torch.sum(torch.mul(losses, qts), 1), 0)
        # Changed the sequence of processing. We already calculate the mean avg loss for each step in the method "step_loss"
        #           there, we already multiplied the losses for each optimizee with the qt-value generated by the RNN
        #           so here we only need to sum over the dim1 which holds the time steps
        losses = torch.sum(losses, 1)
        if self.with_meta_loss:
            self.loss_sum = (losses + kl_term).squeeze() + torch.mean(torch.cat(self.sum_step_losses, 1)).double()
        else:
            self.loss_sum = (losses + kl_term).squeeze()

    def set_qt_values(self, new_probs, next_iteration_condition, final_func_mask):
        qt = Variable(torch.zeros(new_probs.size(0)))
        next_iter = torch.sum(next_iteration_condition).data.cpu().numpy()[0]
        finals = torch.sum(final_func_mask).data.cpu().squeeze().numpy()[0]
        next_idx = next_iteration_condition.data.squeeze().nonzero().squeeze()
        final_idx = final_func_mask.data.squeeze().nonzero().squeeze()
        if new_probs.is_cuda:
            qt = qt.cuda()
        if next_iter > 0:
            qt[next_idx] = new_probs[next_idx]
        # print("{} step next/final {}/{}".format(self.step, next, finals))

        if self.step == 0:
            qt_remainder = self.tensor_one.expand_as(new_probs)
        else:
            qt_remainder = self.tensor_one - self.remainder_probs

        if finals > 0:
            qt[final_idx] = qt_remainder[final_idx]
            # print("remainders", qt_remainder.size())
            # print("in final mask? ", final_func_mask[10].data.cpu().numpy()[0])
            # if final_func_mask[10].data.cpu().numpy()[0] == 1:
            #     print("*** yes in final mask")
            #     print("new_prob ", new_probs[10].data.cpu().squeeze().numpy()[0])
            #     print("in qt ", qt[10].data.cpu().squeeze().numpy()[0])
            #     print("Remainder ", qt_remainder[10].data.cpu().squeeze().numpy()[0])

        self.qts[:, self.step] = qt
        # if self.step == 0:
        #    pass
        #    print(self.qts[10, self.step].data.cpu().squeeze().numpy())
        # else:
        #    print(self.qts[10, 0:self.step+1].data.cpu().squeeze().numpy())

    def construct_priors(self):
        # construct array of arrays with different length of ranges, we need this to construct a 2D matrix that will be
        # passed to the geometric PMF function of scipy
        R = np.array([np.arange(1, i+1) for i in self.halting_steps.data.cpu().numpy()])
        R = np.vstack([np.lib.pad(a, (0, (self.step - len(a))), 'constant', constant_values=0) for a in R])
        if self.type_prior == "geometric":
            g_priors = geom.pmf(R, p=(1-self.prior_shape_param1))
        elif self.type_prior == "neg-binomial":
            g_priors = nbinom.pmf(R, self.prior_shape_param2, p=(1-self.prior_shape_param1))
        else:
            raise ValueError("Unknown prior distribution {}. Only 1) geometric and 2) neg-binomial "
                             "are supported".format(self.type_prior))

        # create truncated priors
        g_priors *= 1./np.sum(g_priors, 1).reshape((self.batch_size, 1))
        g_priors = Variable(torch.from_numpy(g_priors).double())
        if self.qts.is_cuda:
            g_priors = g_priors.cuda()
        return g_priors

    def process_step0(self, exper, loss):
        baseline_loss = torch.mean(loss, 0).data.cpu().squeeze().numpy()[0]
        exper.add_step_loss(baseline_loss, self.step, is_train=self.is_train)
        exper.add_opt_steps(self.step, is_train=self.is_train)







