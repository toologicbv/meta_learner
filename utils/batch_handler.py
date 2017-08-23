import sys
import abc
import numpy as np
from scipy.stats import geom, nbinom
import torch
from torch.autograd import Variable


from utils.common import get_batch_functions, get_func_loss
from utils.helper import tensor_and, tensor_any


from models.rnn_optimizer import get_step_loss


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
        self.learner = exper.args.learner

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
        self.tensor_one = Variable(torch.ones(1).double())
        self.max_T = Variable(torch.FloatTensor([self.horizon-1]).expand_as(self.float_mask))
        if self.is_train:
            self.one_minus_eps = torch.FloatTensor(self.max_T.size()).uniform_(0., 1.).double()
            self.one_minus_eps = Variable(self.tensor_one.data.expand_as(self.one_minus_eps) - self.one_minus_eps)
        else:
            # during evaluation we fix the threshold
            self.one_minus_eps = Variable(torch.zeros(self.max_T.size()).double())
            self.one_minus_eps[:] = exper.config.qt_threshold
        # IMPORTANT, this tensor needs to have the same size as exper.epoch_stats["opt_step_hist"][exper.epoch] which
        # is set in train_optimizer.py in the beginning of the epoch
        self.halting_steps = Variable(torch.zeros(self.max_T.size()))
        self.counter_compare = Variable(torch.zeros(self.max_T.size()))

        self.q_t = Variable(torch.zeros(self.batch_size, self.horizon).double())
        # array of rho_t values that we'll need to calculate the qt-values qt=\prod_{i=1}{t-1} (1-rho_i) rho_t
        self.rho_t = Variable(torch.zeros(self.batch_size, self.horizon).double())
        # IMPORTANT: we're using compare_probs only for comparison in the WHILE loop in order to determine when to stop
        self.compare_probs = Variable(torch.zeros(self.max_T.size()).double())
        self.cumulative_probs = Variable(torch.zeros(self.max_T.size()).double())
        self.time_step = 0
        self.step = 0
        self.loss_sum = 0
        self.kl_term = 0
        self.batch_step_losses = []
        self.backward_ones = torch.ones(self.batch_size)
        # only used during evaluation to capture the last time step when at least one optimizee still needed processing
        self.eval_last_step_taken = 0.
        self.eps = 1e-320
        self.verbose = False
        if exper.args.cuda:
            self.cuda()

    def cuda(self):
        self.bool_mask = self.bool_mask.cuda()
        self.float_mask = self.float_mask.cuda()
        self.max_T = self.max_T.cuda()
        self.one_minus_eps = self.one_minus_eps.cuda()
        self.halting_steps = self.halting_steps.cuda()
        self.counter_compare = self.counter_compare.cuda()
        self.q_t = self.q_t.cuda()
        self.rho_t = self.rho_t.cuda()
        self.compare_probs = self.compare_probs.cuda()
        self.cumulative_probs = self.cumulative_probs.cuda()
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
            delta_param, rho_probs = meta_optimizer.forward(self.functions.params.grad.view(-1))
        else:
            flat_grads = Variable(self.functions.params.grad.data.view(-1))
            delta_param, rho_probs = meta_optimizer.forward(flat_grads)
        # (1) reshape parameter tensor (2) take mean to compute qt values
        delta_param = delta_param.view(param_size)
        rho_probs = torch.mean(rho_probs.view(*param_size), 1, keepdim=True)
        if not self.is_train:
            # during evaluation we keep ALL parameters generated in order to be able to compute new loss values
            # for all optimizees
            eval_par_new = Variable(self.functions.params.data - delta_param.data)
        # then, apply the previous batch mask, although we did the forward pass with all functions, we filter here
        delta_param = torch.mul(delta_param, self.float_mask)
        # moved to method >>> compute_probs <<<
        # rho_new = torch.mul(rho_probs, self.float_mask)
        # compute new probability values, based on the cumulative probs construct new batch mask
        # note that we also save the new rho_t values in the array self.rho_t (in the method compute_probs)
        new_probs = self.compute_probs(rho_probs)
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
        self.compare_probs += torch.mul(new_probs, self.float_mask.double())
        par_new = self.functions.params - delta_param
        # generate object that holds time_step_condition (which function has taken more than MAX time steps allowed)
        time_step_condition = torch.le(self.counter_compare, self.max_T)
        # we then generate the mask that determines which functions will finally be part of the next round
        next_iteration_condition = tensor_and(new_batch_mask, time_step_condition)
        final_func_mask = (self.bool_mask - next_iteration_condition) == 1
        # set the cumulative_probs for all functions that participate in the next time step
        # Subtlety here: so for an optimizee stopping in this step, we don't want the cum-prob to be increased
        #                because we'll use the previous (step-1) cum-probs to determine the q_T value...the final
        #                q_t value for the halting step which gets assigned the REST prob mass
        # we use this object for determining the rest-probability for the functions that stop after this time step
        # the mask "final_func_mask" holds the indices of the functions that stop after this time step
        # the mask "next_iteration_condition" holds the indices of the functions that continue in the next time step
        self.cumulative_probs += torch.mul(new_probs, next_iteration_condition.double())
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

        exper.add_step_qts(self.q_t[:, 0:self.step].data.cpu().numpy(), is_train=self.is_train)
        exper.add_halting_steps(self.halting_steps, is_train=self.is_train)
        # set the class variable if we reached a new maximum time steps
        if epoch_obj.get_max_time_steps_taken(self.is_train) < self.step:
            epoch_obj.set_max_time_steps_taken(self.step, self.is_train)
        if not self.is_train:
            if self.eval_last_step_taken == 0:
                self.eval_last_step_taken = self.step
            epoch_obj.set_max_time_steps_taken(self.eval_last_step_taken, self.is_train)
            exper.meta_logger.info("! - Validation last step {} - !".format(self.eval_last_step_taken))

    def compute_probs(self, new_rho_t):

        # if this is the first time we broke the stick, qts are just our probs
        if self.step == 0:
            probs = new_rho_t.double()
        else:
            # stick-breaking procedure: the new probs = \prod_{i=1}^{t-1} (1 - rho_i) rho_t
            # actually we transform the product into a sum of logs and transform back to probs with torch.exp for
            # numerical stability
            one_minus_rho = torch.log((self.tensor_one.expand_as(new_rho_t) - self.rho_t[:, 0:self.step]))
            probs = torch.exp(torch.sum(one_minus_rho, 1, keepdim=True) + torch.log(new_rho_t.double() + self.eps))
            # previous compute style -- archive
            # probs = torch.mul(torch.prod(self.tensor_one.expand_as(new_rho_t) - self.rho_t[:, 0:self.step], 1,
            #                             keepdim=True), new_rho_t.double())
            probs = torch.mul(probs, self.float_mask.double())

        self.rho_t[:, self.step] = torch.mul(new_rho_t, self.float_mask).squeeze().double()

        return probs

    def backward(self, epoch_obj, meta_optimizer, optimizer, loss_sum=None):
        if len(self.batch_step_losses) == 0:
            raise RuntimeError("No batch losses accumulated. Can't execute backward() on object")
        if loss_sum is None:
            self.compute_batch_loss(kl_weight=epoch_obj.kl_weight)
        else:
            self.loss_sum = loss_sum

        self.loss_sum.backward()
        optimizer.step()
        # print("Sum grads {:.4f}".format(meta_optimizer.sum_grads(verbose=True)))
        meta_optimizer.reset_final_loss()
        sum_grads = meta_optimizer.sum_grads()
        meta_optimizer.zero_grad()
        return self.loss_sum.data.cpu().squeeze().numpy()[0], sum_grads

    def step_loss(self, new_parameters, average_batch=True):

        # Note: for the ACT step loss, we're only summing over the number of samples (dim1), for META model
        # we also sum over dim0 - the number of functions. But for ACT we need the losses per function in the
        # final_loss calculation (multiplied with the qt values, which we collect also for each function
        loss = get_step_loss(self.functions, new_parameters, avg_batch=False)
        if loss.dim() == 1:
            loss = loss.unsqueeze(1)

        self.batch_step_losses.append(loss)
        if average_batch:
            return torch.mean(loss)
        else:
            return loss

    def compute_batch_loss(self, kl_weight=1.):
        # construct the individual priors for each batch function, return DoubleTensor[batch_size, self.steps]
        g_priors = self.construct_priors_v2()
        # get the q_t value for all optimizees for their halting step. NOTE: halting step has to be decreased with
        # ONE because the index of the self.q_t array starts with 0 right
        idx_last_step = Variable(self.halting_steps.data.type(torch.LongTensor) - 1)
        if g_priors.cuda:
            idx_last_step = idx_last_step.cuda()
        q_T_values = torch.gather(self.q_t, 1, idx_last_step)
        # q_T_values = self.compute_last_qt(idx_last_step)
        # compute KL divergence term, take the mean over the mini-batch dimension 0
        kl_term = kl_weight * torch.mean(self.approximate_kl_div(q_T_values, g_priors), 0)
        # get the loss value for each optimizee for the halting step
        loss_matrix = torch.cat(self.batch_step_losses, 1)
        losses = torch.mean(torch.gather(loss_matrix, 1, idx_last_step), 0)
        # compute final loss, in which we multiply each loss by the qt time step values
        # self.loss_sum = (losses.double() + kl_term).squeeze()
        self.loss_sum = kl_term.squeeze()
        self.kl_term = kl_term.data.cpu().squeeze().numpy()[0]

    def compute_last_qt(self, halting_idx):
        qt = Variable(torch.zeros(self.rho_t.size(0)))
        if self.rho_t.cuda:
            qt = qt.cuda()
        i = 0
        for idx in halting_idx.data.cpu().squeeze().numpy():
            if int(idx) == 0:
                qt[i] = self.rho_t[i, int(idx)] + (self.tensor_one - self.rho_t[i, int(idx)])
            else:
                qt[i] = torch.prod(self.tensor_one - self.rho_t[i, 0:int(idx)], 0, keepdim=True)
            i += 1
        return qt.unsqueeze(1)

    def set_qt_values(self, new_probs, next_iteration_condition=None, final_func_mask=None):
        qt = Variable(torch.zeros(new_probs.size(0)).double())
        if new_probs.is_cuda:
            qt = qt.cuda()
        final_due_to_fixed_horizon = torch.eq(self.counter_compare, self.max_T)
        num_of_finals = torch.sum(final_due_to_fixed_horizon).data.cpu().squeeze().numpy()[0]
        final_idx_due_to_fixed_horizon = final_due_to_fixed_horizon.data.squeeze().nonzero().squeeze()
        qt[:] = new_probs
        # print("{} step next/final {}/{}".format(self.step, next, finals))
        if num_of_finals > 0:
            # subtlety here: we compute \prod_{i=1}^{t-1} (1 - rho_i) = remainder.
            #                indexing starts at 0 = first step. self.step starts counting at 0
            #                so when self.step=1 (we're actually in step 2 then) we only compute (1-rho_1)
            #                which should be correct in step 2 (because step 1 = rho_1)
            qt_remainder = self.tensor_one.expand_as(self.compare_probs) - self.compare_probs
            # qt_remainder = torch.prod(self.tensor_one.expand_as(qt) - self.rho_t[:, 0:self.step], 1, keepdim=True)
            qt[final_idx_due_to_fixed_horizon] = (new_probs + qt_remainder)[final_idx_due_to_fixed_horizon]
            if self.verbose:
                print("remainders", qt_remainder.size())
                print("in final mask? ", final_due_to_fixed_horizon[10].data.cpu().numpy()[0])
                if final_due_to_fixed_horizon[10].data.cpu().numpy()[0] == 1:
                    print("*** yes in final mask")
                    print("new_prob ", new_probs[10].data.cpu().squeeze().numpy()[0])
                    print("Remainder ", qt_remainder[10].data.cpu().squeeze().numpy()[0])

        self.q_t[:, self.step] = qt
        if self.verbose and num_of_finals > 0 and final_due_to_fixed_horizon[10].data.cpu().numpy()[0] == 1:
            print(self.q_t[10, 0:self.step + 1].data.cpu().squeeze().numpy())
            print("Sum probs: ", np.sum(self.q_t[10, 0:self.step + 1].data.cpu().squeeze().numpy()))

    def construct_priors_v2(self):
        """
        Just get the p(t) values for the mini-batch using the indices of the "halting step" vector.
        Note: the prior is not truncated
        :return: prior values for optimizees at time step = halting step
        """

        if self.type_prior == "geometric":
            g_priors = geom.pmf(self.halting_steps.data.cpu().numpy(), p=(1-self.prior_shape_param1))
            # g_priors = nbinom.pmf(self.halting_steps.data.cpu().numpy(), 50, p=0.3)
        else:
            raise ValueError("Unknown prior distribution {}. Only 1) geometric and 2) neg-binomial "
                             "are supported".format(self.type_prior))
        g_priors = Variable(torch.from_numpy(g_priors).double())
        if self.rho_t.is_cuda:
            g_priors = g_priors.cuda()
        return g_priors

    def construct_priors_v1(self):
        # construct array of arrays with different length of ranges, we need this to construct a 2D matrix that will be
        # passed to the geometric PMF function of scipy
        R = np.array([np.arange(1, i+1) for i in self.halting_steps.data.cpu().numpy()])
        R = np.vstack([np.lib.pad(a, (0, (self.step - len(a))), 'constant', constant_values=0) for a in R])
        if self.type_prior == "geometric":
            g_priors = geom.pmf(R, p=(1-self.prior_shape_param1))
        else:
            raise ValueError("Unknown prior distribution {}. Only 1) geometric and 2) neg-binomial "
                             "are supported".format(self.type_prior))

        g_priors = Variable(torch.from_numpy(g_priors).double())
        if self.q_t.is_cuda:
            g_priors = g_priors.cuda()
        return g_priors

    def process_step0(self, exper, loss):
        baseline_loss = torch.mean(loss, 0).data.cpu().squeeze().numpy()[0]
        exper.add_step_loss(baseline_loss, self.step, is_train=self.is_train)
        exper.add_opt_steps(self.step, is_train=self.is_train)

    def approximate_kl_div(self, q_probs, prior_probs):

        try:
            kl_div = torch.sum(torch.log(q_probs + self.eps) - torch.log(prior_probs + self.eps), 1)
        except RuntimeError:
            print("q_probs.size ", q_probs.size())
            print("prior_probs.size ", prior_probs.size())
            raise RuntimeError("Running away from here...")

        return kl_div


class ACTGravesBatchHandler(ACTBatchHandler):

    def __init__(self, exper, is_train, optimizees=None):
        super(ACTGravesBatchHandler, self).__init__(exper, is_train, optimizees)

        self.one_minus_eps[:] = exper.config.qt_threshold

    def compute_batch_loss(self, kl_weight=1.):
        ponder_cost = kl_weight * self.compute_ponder_cost()
        loss_matrix = torch.cat(self.batch_step_losses, 1)
        qts = self.q_t[:, 0:loss_matrix.size(1)]
        losses = torch.mean(torch.sum(torch.mul(qts, loss_matrix), 1), 0)
        self.loss_sum = (losses.double() + ponder_cost).squeeze()
        self.kl_term = ponder_cost.data.cpu().squeeze().numpy()[0]

    def compute_ponder_cost(self):
        R = np.array([np.arange(1, i + 1) for i in self.halting_steps.data.cpu().numpy()])
        R = np.vstack([np.lib.pad(a, (0, (self.horizon - len(a))), 'constant', constant_values=0) for a in R])
        R = Variable(torch.from_numpy(R).double())
        if self.q_t.is_cuda:
            R = R.cuda()

        C = torch.mul(R, self.q_t)
        return torch.mean(torch.sum(C, 1), 0)

    def set_qt_values(self, new_probs, next_iteration_condition, final_func_mask):
        qt = Variable(torch.zeros(new_probs.size(0)).double())
        next_iter = torch.sum(next_iteration_condition).data.cpu().numpy()[0]
        finals = torch.sum(final_func_mask).data.cpu().squeeze().numpy()[0]
        next_idx = next_iteration_condition.data.squeeze().nonzero().squeeze()
        final_idx = final_func_mask.data.squeeze().nonzero().squeeze()
        if new_probs.is_cuda:
            qt = qt.cuda()
        if next_iter > 0:
            qt[next_idx] = new_probs[next_idx]
        # print("{} step next/final {}/{}".format(self.step, next, finals))
        if finals > 0:
            # subtlety here: we compute \prod_{i=1}^{t-1} (1 - rho_i) = remainder.
            #                indexing starts at 0 = first step. self.step starts counting at 0
            #                so when self.step=1 (we're actually in step 2 then) we only compute (1-rho_1)
            #                which should be correct in step 2 (because step 1 = rho_1)
            qt_remainder = self.tensor_one.expand_as(self.compare_probs) - self.compare_probs
            # qt_remainder = torch.prod(self.tensor_one.expand_as(qt) - self.rho_t[:, 0:self.step], 1, keepdim=True)
            qt[final_idx] = (new_probs + qt_remainder)[final_idx]
            if self.verbose:
                print("remainders", qt_remainder.size())
                print("in final mask? ", final_func_mask[10].data.cpu().numpy()[0])
                if final_func_mask[10].data.cpu().numpy()[0] == 1:
                    print("*** yes in final mask")
                    print("new_prob ", new_probs[10].data.cpu().squeeze().numpy()[0])
                    print("in qt ", qt[10].data.cpu().squeeze().numpy()[0])
                    print("Remainder ", qt_remainder[10].data.cpu().squeeze().numpy()[0])

        self.q_t[:, self.step] = qt
        if self.verbose and finals > 0 and final_func_mask[10].data.cpu().numpy()[0] == 1:
            print(self.q_t[10, 0:self.step + 1].data.cpu().squeeze().numpy())
            print("Sum probs: ", np.sum(self.q_t[10, 0:self.step + 1].data.cpu().squeeze().numpy()))


class ACTEfficientBatchHandler(ACTBatchHandler):

    def __init__(self, exper, is_train, optimizees=None):
        super(ACTEfficientBatchHandler, self).__init__(exper, is_train, optimizees)
        self.verbose = False

    def compute_batch_loss(self, kl_weight=1.):
        # construct the individual priors for each batch function, return DoubleTensor[batch_size, self.steps]
        g_priors = self.construct_priors_v1()
        q_T_values = self.q_t[:, 0:self.step].double()
        # compute KL divergence term, take the mean over the mini-batch dimension 0
        kl_term = kl_weight * torch.mean(torch.sum(torch.mul(self.rho_t[:, 0:self.step],
                                                             self.approximate_kl_div(q_T_values, g_priors))
                                                   , 1)
                                         , 0)
        # get the loss value for each optimizee for the halting step
        loss_matrix = torch.cat(self.batch_step_losses, 1).double()
        losses = torch.mean(torch.sum(torch.mul(q_T_values, loss_matrix), 1), 0)
        # compute final loss, in which we multiply each loss by the qt time step values
        # self.loss_sum = (losses + kl_term).squeeze()
        self.loss_sum = kl_term.squeeze()
        self.kl_term = kl_term.data.cpu().squeeze().numpy()[0]

    def approximate_kl_div(self, q_probs, prior_probs, verbose=False):
        """
        NOTE: only difference with the same method from parent class is that we're NOT taking the sum over the
         time steps here, because we are going to multiply each kl-term(t) with the appropriate rho_t value
        :param q_probs:
        :param prior_probs:
        :return:
        """
        # we need a ByteTensor mask because q_probs has size [batch_size, self.step] and contains zeros for all
        # steps of an optimizee after the halting step. Hence we need to multiply the result by the mask, passed as
        # double()
        mask = q_probs > 0.
        try:
            kl_div = torch.mul(torch.log(q_probs + self.eps) - torch.log(prior_probs + self.eps), mask.double())
            if verbose:
                max_steps = torch.sum(q_probs[0] > 0).data.cpu().numpy()[0]
                if max_steps > 30:
                    print("q_probs length {}".format(max_steps))
                    print(q_probs[0].data.cpu().squeeze().numpy())
                    print(prior_probs[0].data.cpu().squeeze().numpy())
                    print("Sum KL-div for optimizee[0]")
                    print(np.sum(kl_div[0].data.cpu().squeeze().numpy()))
        except RuntimeError:
            print("q_probs.size ", q_probs.size())
            print("prior_probs.size ", prior_probs.size())
            raise RuntimeError("Running away from here...")

        return kl_div


