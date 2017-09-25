import sys
import abc
import numpy as np
from scipy.stats import geom, nbinom, poisson
import torch
from torch.autograd import Variable

from utils.helper import preprocess_gradients, get_step_loss
from utils.common import get_batch_functions, get_func_loss
from utils.helper import tensor_and, tensor_any, LessOrEqual


class BatchHandler(object):
    __metaclass__ = abc.ABCMeta

    # static class variable to count the batches
    id = 0

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

    def __init__(self, exper, is_train, optimizees=None):

        self.is_train = is_train
        self.type_prior = exper.type_prior
        self.prior_shape_param1 = exper.config.ptT_shape_param
        self.learner = exper.args.learner
        # TODO temporary added self.version for test puposes different batch loss computation
        self.version = exper.args.version

        if self.is_train:
            self.functions = get_batch_functions(exper)
            self.horizon = exper.config.T
        else:
            self.horizon = exper.config.max_val_opt_steps
            if optimizees is None:
                raise ValueError("Parameter -optimizees- can't be None. Please provide a test set")
            self.functions = optimizees

        self.func_is_nn_module = torch.nn.Module in self.functions.__class__.__bases__
        self.batch_size = self.functions.num_of_funcs
        # will be intensively used to select the functions that still need to be optimzed after t steps
        self.bool_mask = Variable(torch.ones(self.batch_size, 1).type(torch.ByteTensor))
        self.float_mask = Variable(torch.ones(self.batch_size, 1))
        self.tensor_one = Variable(torch.ones(1).double())
        self.max_T = Variable(torch.FloatTensor([self.horizon]).expand_as(self.float_mask))
        if self.is_train:
            self.one_minus_eps = torch.FloatTensor(self.max_T.size()).uniform_(0, 1.).double()
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
        self.qt_last_step = Variable(torch.zeros(self.max_T.size()).double())
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
        self.iterations = Variable(torch.zeros(self.batch_size, 1))
        self.qt_remainders = Variable(torch.zeros(self.batch_size, 1).double())
        self.penalty_term = 0.
        self.test_result_scores = []
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
        self.qt_last_step = self.qt_last_step.cuda()
        self.iterations = self.iterations.cuda()
        self.qt_remainders = self.qt_remainders.cuda()

    def act_step(self, exper, meta_optimizer):
        """
        Subtleties of the batch processing. TODO: NEED TO EXPLAIN THIS!!!
        :param exper:
        :param meta_optimizer:
        :return: IMPORTANT RETURNS LOSS STEP THAT IS NOT MULTIPLIED BY QT-VALUE! NUMPY FLOAT32
        """
        loss = get_func_loss(exper, self.functions, average=False)
        # make the forward step, which depends heavily on the experiment we're performing (MLP or Regression(T))
        delta_param, rho_probs, eval_par_new = self.forward(loss, exper, meta_optimizer)
        # then, apply the previous batch mask, although we did the forward pass with all functions, we filter here
        delta_param = torch.mul(delta_param, self.float_mask)
        # moved to method >>> compute_probs <<<
        # rho_new = torch.mul(rho_probs, self.float_mask)
        # compute new probability values, based on the cumulative probs construct new batch mask
        # note that we also save the new rho_t values in the array self.rho_t (in the method compute_probs)
        if self.learner == "meta_act":
            # in Graves ACT model, the qt values are probabilities already : sigmoid(W^T h_t + bias) values
            # so we don't use any stick-breaking here (meaning the rho_t values that we transfer in the act_sb
            new_probs = torch.mul(rho_probs.double(), self.float_mask.double())
        else:
            # stick-breaking approach: transform RNN output rho_t values to probabilities
            # method compute_probs multiplies with the float_mask object!
            new_probs = self.compute_probs(rho_probs)
        # we need to determine the indices of the functions that "stop" in this time step (new_funcs_mask)
        funcs_that_stop = torch.le(self.compare_probs + new_probs, self.one_minus_eps)
        less_or_equal = LessOrEqual()
        # NOTE: ALSO for the Graves ACT model we increase all functions that participated in THIS ITERATION and
        #       therefore we compare with self.compare_probs BEFORE we increase them with the new probs
        iterations = less_or_equal(self.compare_probs, self.one_minus_eps)
        if self.learner == "meta_act":
            self.halting_steps += iterations
        else:
            # and we increase the number of time steps TAKEN with the previous float_mask object to keep track of the steps
            self.halting_steps += self.float_mask
            self.iterations += iterations

        new_batch_mask = tensor_and(funcs_that_stop, self.bool_mask)
        new_float_mask = new_batch_mask.float()

        # IMPORTANT: although above we constructed the new mask (for the next time step) we use the previous mask for the
        # increase of the cumulative probs which we use in the WHILE loop to determine when to stop the complete batch
        self.compare_probs += torch.mul(new_probs, self.float_mask.double())
        # IMPORTANT UPDATE OF OPTIMIZEE PARAMETERS
        if self.func_is_nn_module:
            # awkward but true: delta_param has [batch_dim, num_params] due to multiplication with float masks, but
            # must have transpose shape for this update
            par_new = self.functions.get_flat_params() - delta_param.permute(1, 0)
        else:
            par_new = self.functions.params - delta_param
        # increase the number of steps taken for the functions that are still in the race, we need these to compare
        # against the maximum allowed number of steps
        self.counter_compare += new_float_mask
        # generate object that holds time_step_condition (which function has taken more than MAX time steps allowed)
        # Note that self.max_T = HORIZON
        time_step_condition = torch.lt(self.counter_compare, self.max_T)
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
            loss_step = self.step_loss(par_new, exper, average_batch=True)
        else:
            loss_step = self.step_loss(eval_par_new, exper, average_batch=True)
        # update batch functions parameter for next step
        if self.is_train:
            self.functions.set_parameters(par_new)
        else:
            self.functions.set_parameters(eval_par_new)

        if self.func_is_nn_module:
            self.functions.zero_grad()
        else:
            self.functions.params.grad.data.zero_()

        return loss_step.data.cpu().squeeze().numpy()[0]

    def forward(self, loss, exper, meta_optimizer):
        # Note: splitting the logic between the different experiments:
        #       (1) MLP
        #       (2) All others (Regression & Regression_T)
        # compute gradients of optimizee which will need for the meta-learner
        if exper.args.problem == "mlp":
            loss.backward()
            if self.is_train:
                delta_param, rho_probs = meta_optimizer.forward(Variable(torch.cat((preprocess_gradients(
                    self.functions.get_flat_grads().data),
                    self.functions.get_flat_params().data), 1)))
            else:
                delta_param, rho_probs = meta_optimizer.forward(Variable(torch.cat((preprocess_gradients(
                    self.functions.get_flat_grads().data),
                    self.functions.get_flat_params().data),
                    1), volatile=True))

            if not self.is_train:
                # during evaluation we keep ALL parameters generated in order to be able to compute new loss values
                # for all optimizees
                eval_par_new = Variable(self.functions.get_flat_params().data - delta_param.data.unsqueeze(1))
            else:
                eval_par_new = None
            # we have no batch dimension, so we add one in order to adjust to the
            # rho_probs has shape [num_of_flat_parameters, ]. Note, batch dimension is dim0
            # for the rho_probs we can just add the 2nd dim where ever we want, because we take the mean anyway [1 x 1]
            # for the delta_param the situation is unfortunately more confusing. In the act_step method [1 x num-params]
            # would be appreciated because of the float-masks which have batch dim as dim0. But the set_parameter method
            # of MLP object expects [num_params, 1]...
            delta_param = delta_param.unsqueeze(0)
            rho_probs = rho_probs.unsqueeze(1)
            rho_probs = torch.mean(rho_probs, 0, keepdim=True)

        else:
            loss.backward(self.backward_ones)
            param_size = self.functions.params.grad.size()
            if self.is_train:
                flat_grads = Variable(self.functions.params.grad.data.view(-1))
                delta_param, rho_probs = meta_optimizer.forward(flat_grads)
            else:
                # IMPORTANT! BECAUSE OTHERWISE RUNNING INTO MEMORY ISSUES - PASS GRADS NOT IN A NEW VARIABLE
                delta_param, rho_probs = meta_optimizer.forward(self.functions.params.grad.view(-1))

            # (1) reshape parameter tensor (2) take mean to compute qt values
            # try to produce ones
            delta_param = delta_param.view(param_size)
            rho_probs = torch.mean(rho_probs.view(*param_size), 1, keepdim=True)

            if not self.is_train:
                # during evaluation we keep ALL parameters generated in order to be able to compute new loss values
                # for all optimizees
                eval_par_new = Variable(self.functions.params.data - delta_param.data)
            else:
                eval_par_new = None
        # register the baseline loss at step 0
        if self.step == 0:
            self.process_step0(exper, loss)

        return delta_param, rho_probs, eval_par_new

    def __call__(self, exper, epoch_obj, meta_optimizer, final_batch=False):

        self.step = 0
        meta_optimizer.reset_lstm(keep_states=False)
        if self.is_train:
            do_continue = tensor_any(tensor_and(torch.le(self.compare_probs, self.one_minus_eps),
                                                torch.lt(self.counter_compare, self.max_T))).data.cpu().numpy()[0]
        else:
            do_continue = True if self.step < self.horizon-1 else False

        while do_continue:
            # IMPORTANT! avg_loss_step is NOT MULTIPLIED BY THE qt-value!!!!
            avg_loss_step = self.act_step(exper, meta_optimizer)
            if self.is_train:
                do_continue = tensor_any(
                    tensor_and(torch.le(self.compare_probs, self.one_minus_eps),
                               torch.lt(self.counter_compare, self.max_T))).data.cpu().numpy()[0]
            else:
                do_continue = True if self.step < self.horizon-1 else False
                if self.eval_last_step_taken == 0:
                    num_next_iter = torch.sum(self.float_mask).data.cpu().squeeze().numpy()[0]
                    if num_next_iter == 0. and self.eval_last_step_taken == 0.:
                        self.eval_last_step_taken = self.step + 1

            # important increase step after "do_continue" stuff but before adding step losses
            self.step += 1
            if self.learner == "act_sb_eff" and self.is_train:
                # evaluating the lower bound after each step
                self.backward(epoch_obj, meta_optimizer, exper.optimizer, retain_graph=True)
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

        if exper.args.problem == "mlp" and final_batch:
            # evaluate the last MLP that we optimized
            accuracy = self.functions.test_model(exper.dta_set, exper.args.cuda, quick_test=True)
            # exper.meta_logger.info("Epoch {}: last batch - accuracy of last MLP {:.4f}".format(exper.epoch, accuracy))
            self.test_result_scores.append(accuracy)

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

    def backward(self, epoch_obj, meta_optimizer, optimizer, loss_sum=None, retain_graph=False):
        if len(self.batch_step_losses) == 0:
            raise RuntimeError("No batch losses accumulated. Can't execute backward() on object")
        if loss_sum is None:
            self.compute_batch_loss(weight_regularizer=epoch_obj.weight_regularizer)
        else:
            self.loss_sum = loss_sum

        self.loss_sum.backward(retain_graph=retain_graph)
        # sum_grads = 0
        # if meta_optimizer.rho_linear_out.weight.grad is not None:
        #     sum_grads += torch.sum(meta_optimizer.rho_linear_out.weight.grad.data)
        # else:
        #     print(">>>>>>>>>>>>> NO GRADIENTS <<<<<<<<<<<<")
        # if meta_optimizer.rho_linear_out.bias.grad is not None:
        #     sum_grads += torch.sum(meta_optimizer.rho_linear_out.bias.grad.data)
        # print("Sum rho layer gradients ", sum_grads)
        optimizer.step()
        # print("Sum grads {:.4f}".format(meta_optimizer.sum_grads(verbose=True)))
        # we don't want to reset all our internal meta learner variables when using efficient sampling
        if not retain_graph:
            meta_optimizer.reset_final_loss()
        sum_grads = meta_optimizer.sum_grads()
        meta_optimizer.zero_grad()
        return self.loss_sum.data.cpu().squeeze().numpy()[0], sum_grads

    def step_loss(self, new_parameters, exper, average_batch=True):

        # Note: for the ACT step loss, we're only summing over the number of samples (dim1), for META model
        # we also sum over dim0 - the number of functions. But for ACT we need the losses per function in the
        # final_loss calculation (multiplied with the qt values, which we collect also for each function
        loss = get_step_loss(self.functions, new_parameters, avg_batch=False, exper=exper, is_train=self.is_train)
        if loss.dim() == 1:
            loss = loss.unsqueeze(1)

        self.batch_step_losses.append(loss)
        if average_batch:
            return torch.mean(loss)
        else:
            return loss

    def compute_batch_loss(self, weight_regularizer=1., variational=True, original_kl=False, mean_field=True):

        # get the q_t value for all optimizees for their halting step. NOTE: halting step has to be decreased with
        # ONE because the index of the self.q_t array starts with 0 right
        if self.version == "V3.1" or self.version == 'V3.2':
            original_kl = False
            variational = True
            mean_field = True
        elif self.version == "V1":
            variational = True
            original_kl = True
            mean_field = False

        idx_last_step = Variable(self.halting_steps.data.type(torch.LongTensor) - 1)
        if self.q_t.cuda:
            idx_last_step = idx_last_step.cuda()
        q_T_values = torch.gather(self.q_t, 1, idx_last_step)
        # q_T_values = self.compute_last_qt(idx_last_step)
        # compute KL divergence term, take the mean over the mini-batch dimension 0
        if variational:
            if original_kl:
                # construct the individual priors for each batch function, return DoubleTensor[batch_size, self.steps]
                g_priors = self.construct_priors_v2()
                kl_term = weight_regularizer * torch.mean(self.approximate_kl_div(q_T_values, g_priors), 0)
            else:
                # construct the individual priors for each batch function, return DoubleTensor[batch_size, self.steps]
                if self.version == 'V3.2':
                    g_priors = self.construct_priors_v1(truncate=True)
                else:
                    g_priors = self.construct_priors_v1()
                qts = self.q_t[:, 0:self.step].double()
                # compute KL divergence term, take the mean over the mini-batch dimension 0
                kl_term = weight_regularizer * torch.mean(torch.sum(torch.mul(qts,
                                                                              self.approximate_kl_div_with_sum(qts, g_priors))
                                                                    , 1)
                                                          , 0)
        else:
            kl_term = weight_regularizer * self.compute_stochastic_ponder_cost()

        # get the loss value for each optimizee for the halting step
        if mean_field:
            # get the loss value for each optimizee for the halting step
            loss_matrix = torch.cat(self.batch_step_losses, 1).double()
            # last_step_loss = torch.mean(torch.gather(loss_matrix, 1, idx_last_step), 0)
            losses = torch.mean(torch.sum(torch.mul(self.q_t[:, 0:self.step], loss_matrix), 1), 0) # + last_step_loss
        else:
            loss_matrix = torch.cat(self.batch_step_losses, 1)
            # REMEMBER THIS IS act_sbV2 where we multiply q_t with the log-likelihood
            # losses = torch.mean(torch.sum(torch.mul(q_T_values, loss_matrix.double()), 1), 0)
            # and this is act_sbV1
            losses = torch.mean(torch.gather(loss_matrix, 1, idx_last_step), 0)
        # compute final loss, in which we multiply each loss by the qt time step values
        # remainder = torch.mean(self.iterations).double()
        if self.learner == "act_sb" and (self.version == "V3.1"):
            remainder = weight_regularizer * torch.mean(self.qt_remainders).double()
            self.loss_sum = (losses.double() + remainder).squeeze()
            self.penalty_term = remainder.data.cpu().squeeze().numpy()[0]
            self.kl_term = 0.
        else:
            self.loss_sum = (losses.double() + kl_term).squeeze()

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
        all_finals = torch.sum(final_func_mask).data.cpu().squeeze().numpy()[0]
        if self.version == "V3.2":
            final_due_to_fixed_horizon = final_func_mask
        else:
            final_due_to_fixed_horizon = torch.eq(self.counter_compare, self.max_T)

        num_of_finals = torch.sum(final_due_to_fixed_horizon).data.cpu().squeeze().numpy()[0]
        final_idx_due_to_fixed_horizon = final_due_to_fixed_horizon.data.squeeze().nonzero().squeeze()
        qt[:] = new_probs
        # Note: we only add the remainder of the probability mass for those optimizees that reached the MAX number of
        # time steps condition. In this case we reached the theoretical INFINITE horizon and therefore we make sure the
        # prob-mass adds up to one
        self.qt_remainders = Variable(torch.zeros(self.batch_size, 1).double())
        if new_probs.is_cuda:
            self.qt_remainders = self.qt_remainders.cuda()
        if num_of_finals > 0:
            # subtlety here: we compute \prod_{i=1}^{t-1} (1 - rho_i) = remainder.
            #                indexing starts at 0 = first step. self.step starts counting at 0
            #                so when self.step=1 (we're actually in step 2 then) we only compute (1-rho_1)
            #                which should be correct in step 2 (because step 1 = rho_1)
            self.qt_remainders[final_idx_due_to_fixed_horizon] = (self.tensor_one.expand_as(self.compare_probs) -
                                                                  self.compare_probs)[final_idx_due_to_fixed_horizon]
            # self.qt_remainders = (self.tensor_one.expand_as(self.compare_probs) - self.compare_probs)
            # qt_remainder = torch.prod(self.tensor_one.expand_as(qt) - self.rho_t[:, 0:self.step], 1, keepdim=True)
            qt[final_idx_due_to_fixed_horizon] = (new_probs + self.qt_remainders)[final_idx_due_to_fixed_horizon]

            if self.verbose:
                idx = final_idx_due_to_fixed_horizon[0]
                print("new_prob ", new_probs[idx].data.cpu().squeeze().numpy()[0])
                print("Remainder ", self.qt_remainders[idx].data.cpu().squeeze().numpy()[0])
                print("qt-values")
        if all_finals > 0 and self.version == "V3.1":
            self.qt_remainders = (self.tensor_one.expand_as(self.compare_probs) - self.compare_probs)

        self.q_t[:, self.step] = qt
        if self.verbose and num_of_finals > 0:
            print("self.q_t.size(1) {} and self.step {}".format(self.q_t.size(1), self.step))
            print(self.q_t[idx, -10:].data.cpu().squeeze().numpy())
            print("Sum probs: ", np.sum(self.q_t[idx, 0:self.step + 1].data.cpu().squeeze().numpy()))

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

    def construct_priors_v1(self, truncate=False):
        # construct array of arrays with different length of ranges, we need this to construct a 2D matrix that will be
        # passed to the geometric PMF function of scipy
        R = np.array([np.arange(1, i+1) for i in self.halting_steps.data.cpu().numpy()])
        if self.step > 1:
            R = np.vstack([np.lib.pad(a, (0, (self.step - len(a))), 'constant', constant_values=0) for a in R])
        if self.type_prior == "geometric":
            g_priors = geom.pmf(R, p=(1-self.prior_shape_param1))

        else:
            raise ValueError("Unknown prior distribution {}. Only 1) geometric and 2) neg-binomial "
                             "are supported".format(self.type_prior))

        if truncate:
            g_priors = 1./np.sum(g_priors) * g_priors
        g_priors = Variable(torch.from_numpy(g_priors).double())
        if self.q_t.is_cuda:
            g_priors = g_priors.cuda()
        return g_priors

    def process_step0(self, exper, loss):
        # in the MLP experiments the loss has size [1x1], we don't want to take the mean then
        if loss.size(0) > 1:
            baseline_loss = torch.mean(loss, 0).data.cpu().squeeze().numpy()[0]
        else:
            baseline_loss = loss.data.cpu().squeeze().numpy()[0]
        exper.add_step_loss(baseline_loss, self.step, is_train=self.is_train)
        exper.add_opt_steps(self.step, is_train=self.is_train)

    def compute_stochastic_ponder_cost(self):
        """
        According to Graves paper a possible ponder cost when working with a stochastic ACT approach (see footnote 1
        page 5): \sum_{n=1}^{N(t)} n p_t^{(n)}
        :return:
        """
        R = np.array([np.arange(1, i + 1) for i in self.halting_steps.data.cpu().numpy()])
        R = np.vstack([np.lib.pad(a, (0, (self.horizon - len(a))), 'constant', constant_values=0) for a in R])
        R = Variable(torch.from_numpy(R).double())
        if self.q_t.is_cuda:
            R = R.cuda()

        C = torch.mul(R, self.q_t)
        return torch.mean(torch.sum(C, 1), 0)

    def approximate_kl_div(self, q_probs, prior_probs):

        try:
            kl_div = torch.sum(torch.log(q_probs + self.eps) - torch.log(prior_probs + self.eps), 1)
        except RuntimeError:
            print("q_probs.size ", q_probs.size())
            print("prior_probs.size ", prior_probs.size())
            raise RuntimeError("Running away from here...")

        return kl_div

    def approximate_kl_div_with_sum(self, q_probs, prior_probs, verbose=False):
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


class ACTGravesBatchHandler(ACTBatchHandler):

    def __init__(self, exper, is_train, optimizees=None):
        super(ACTGravesBatchHandler, self).__init__(exper, is_train, optimizees)

        self.one_minus_eps[:] = exper.config.qt_threshold

    def cuda(self):
        super(ACTGravesBatchHandler, self).cuda()

    def compute_batch_loss(self, weight_regularizer=1., loss_type="mean_field"):
        # loss_type: (1) mean_field (2) final_step (3) combined
        ponder_cost = self.compute_ponder_cost(tau=weight_regularizer)
        if loss_type == "mean_field":
            loss_matrix = torch.cat(self.batch_step_losses, 1).double()
            qts = self.q_t[:, 0:loss_matrix.size(1)]
            losses = torch.mean(torch.sum(torch.mul(qts, loss_matrix), 1), 0)

        elif loss_type == "combined":
            loss_matrix = torch.cat(self.batch_step_losses, 1).double()
            qts = self.q_t[:, 0:loss_matrix.size(1)]

            idx_last_step = Variable(self.halting_steps.data.type(torch.LongTensor) - 1)
            if self.halting_steps.cuda:
                idx_last_step = idx_last_step.cuda()
            last_losses = torch.mean(torch.gather(loss_matrix, 1, idx_last_step), 0)

            losses = torch.mean(torch.sum(torch.mul(qts, loss_matrix), 1), 0) + last_losses

        elif loss_type == "final_step":
            idx_last_step = Variable(self.halting_steps.data.type(torch.LongTensor) - 1)
            if self.halting_steps.cuda:
                idx_last_step = idx_last_step.cuda()
            loss_matrix = torch.cat(self.batch_step_losses, 1)
            losses = torch.mean(torch.gather(loss_matrix, 1, idx_last_step), 0)
        else:
            raise ValueError("Parameter loss_type {} not supported by this implementation".format(loss_type))

        self.loss_sum = (losses.double() + ponder_cost).squeeze()
        self.kl_term = ponder_cost.data.cpu().squeeze().numpy()[0]

    def compute_ponder_cost(self, tau=5e-2):
        """
        ponder cost for ONE time sequence according to the Graves ACT paper: c = N(t) + R(t)
        where N(t) is the number of steps taken (in our case the halting step) plus R(t) the remainder of the
        probability = (1 - \sum_{n=1}^{t-1} q_t) in our case

        Important note on tau parameter:
        According to Graves, they conducted grid-search for the synthetic tasks: tau = i x 10-j
         i between 1-10 and j between 1-4 (see page 7)
         :param tau: hyperparameter to scale the cost
        :return:
        """

        c = tau * torch.sum(torch.mean(self.halting_steps).double() + torch.mean(self.qt_last_step))
        return c

    def set_qt_values(self, new_probs, next_iteration_condition, final_func_mask):
        qt = Variable(torch.zeros(new_probs.size()).double())
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
            self.qt_last_step[final_idx] = (new_probs + qt_remainder)[final_idx]
            qt[final_idx] = self.qt_last_step[final_idx]
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

    def compute_batch_loss(self, weight_regularizer=1.):
        # construct the individual priors for each batch function, return DoubleTensor[batch_size, self.steps]
        g_priors = self.construct_priors_v1()
        q_T_values = self.q_t[:, 0:self.step].double()
        # compute KL divergence term, take the mean over the mini-batch dimension 0
        kl_term = weight_regularizer * torch.mean(torch.sum(torch.mul(self.rho_t[:, 0:self.step],
                                                                      self.approximate_kl_div_with_sum(q_T_values, g_priors))
                                                            , 1)
                                                  , 0)
        # get the loss value for each optimizee for the halting step
        loss_matrix = torch.cat(self.batch_step_losses, 1).double()
        losses = torch.mean(torch.sum(torch.mul(self.rho_t[:, 0:self.step], loss_matrix), 1), 0)  # q_T_values
        # compute final loss, in which we multiply each loss by the qt time step values
        self.loss_sum = (losses + kl_term).squeeze()

        self.kl_term = kl_term.data.cpu().squeeze().numpy()[0]


