import abc
import numpy as np
from scipy.stats import geom
import torch
from torch.autograd import Variable


from utils.utils import get_batch_functions, get_func_loss
from utils.helper import tensor_and, tensor_any


from models.rnn_optimizer import get_step_loss, kl_divergence


class Batch(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, exper):
        self.batch_size = exper.args.batch_size

    @abc.abstractmethod
    def cuda(self):
        pass

    @abc.abstractmethod
    def __call__(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def backward(self, *args):
        pass


class ACTBatch(Batch):

    # static class variable to count the batches
    id = 0

    def __init__(self, exper):
        super(ACTBatch, self).__init__(exper)
        # will be intensively used to select the functions that still need to be optimzed after t steps
        self.bool_mask = Variable(torch.ones(self.batch_size, 1).type(torch.ByteTensor))
        self.float_mask = Variable(torch.ones(self.batch_size, 1))
        self.max_T = Variable(torch.FloatTensor([exper.config.T]).expand_as(self.float_mask))
        self.one_minus_eps = torch.FloatTensor(self.max_T.size()).uniform_(0.01, 0.05)
        self.one_minus_eps = Variable(1. - self.one_minus_eps)
        # IMPORTANT, this tensor needs to have the same size as exper.epoch_stats["opt_step_hist"][exper.epoch] which
        # is set in train_optimizer.py in the beginning of the epoch
        self.halting_steps = Variable(torch.zeros(self.max_T.size()))
        self.counter_compare = Variable(torch.zeros(self.max_T.size()))
        #
        self.one_minus_qt = Variable(torch.zeros(self.batch_size, exper.config.T))
        self.qts = Variable(torch.zeros(self.batch_size, exper.config.T))
        # IMPORTANT: we're using cum_qt only for comparison in the WHILE loop in order to determine when to stop
        self.compare_probs = Variable(torch.zeros(self.max_T.size()))
        self.remainder_probs = Variable(torch.zeros(self.max_T.size()))
        self.time_step = 0
        self.functions = get_batch_functions(exper)
        self.step = 0
        self.loss_sum = 0
        self.batch_step_losses = []
        self.tensor_one = Variable(torch.ones(1))
        self.geom_shape_param = exper.config.ptT_shape_param
        self.backward_ones = torch.ones(exper.args.batch_size)

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

    def act_step(self, exper, meta_optimizer, keep_fine_losses=False):
        loss = get_func_loss(exper, self.functions, average=False)
        # compute gradients of optimizee which will need for the meta-learner
        loss.backward(self.backward_ones)
        # register the baseline loss at step 0
        if self.step == 0:
            self.process_step0(exper, loss)
        param_size = self.functions.params.grad.size()
        flat_grads = Variable(self.functions.params.grad.data.view(-1))
        delta_param, delta_qt = meta_optimizer.forward(flat_grads)
        # (1) reshape parameter tensor (2) take mean to compute qt values
        delta_param = delta_param.view(param_size)
        delta_qt = torch.mean(delta_qt.view(*param_size), 1, keepdim=True)
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
        loss_step = self.step_loss(par_new, average_batch=True, keep_fine_losses=keep_fine_losses)
        # update batch functions parameter for next step
        self.functions.set_parameters(par_new)
        self.functions.params.grad.data.zero_()

        return loss_step.data.cpu().squeeze().numpy()[0]

    def __call__(self, exper, epoch_obj, meta_optimizer, is_train=False):

        self.step = 0
        meta_optimizer.reset_lstm(keep_states=False)
        do_continue = tensor_any(tensor_and(torch.le(self.compare_probs, self.one_minus_eps),
                                            torch.le(self.counter_compare, self.max_T))).data.cpu().numpy()[0]
        while do_continue:
            avg_loss_step = self.act_step(exper, meta_optimizer, keep_fine_losses=is_train)
            # le1 = np.sum(torch.le(self.cum_qt, self.one_minus_eps).data.cpu().numpy())
            # le2 = np.sum(torch.le(self.counter_compare, self.max_T).data.cpu().numpy())
            do_continue = tensor_any(
                tensor_and(torch.le(self.compare_probs, self.one_minus_eps),
                           torch.le(self.counter_compare, self.max_T))).data.cpu().numpy()[0]

            self.step += 1
            exper.add_step_loss(avg_loss_step, self.step)
            exper.add_opt_steps(self.step)
            epoch_obj.add_step_loss(avg_loss_step, last_time_step=not do_continue)

        # set the class variable if we reached a new maximum time steps
        if epoch_obj.get_max_time_steps_taken() < self.step:
            epoch_obj.set_max_time_steps_taken(self.step)

    def compute_probs(self, qts):

        # if this is the first time we broke the stick, qts are just our probs
        if self.step == 0:
            probs = qts
        else:
            # stick-breaking procedure: the new probs = \prod_{i=1}^{t-1} (1 - qt_i) qt_t
            probs = torch.mul(torch.prod(self.one_minus_qt[:, 0:self.step], 1, keepdim=True), qts)

        self.one_minus_qt[:, self.step] = self.tensor_one - Variable(qts.data.squeeze())
        return probs

    def backward(self, epoch_obj, meta_optimizer, optimizer, kl_scaler=1.):
        if len(self.batch_step_losses) == 0:
            raise RuntimeError("No batch losses accumulated. Can't execute backward() on object")
        self.compute_batch_loss(kl_scaler=kl_scaler)
        self.loss_sum.backward()
        optimizer.step()
        meta_optimizer.reset_final_loss()
        meta_optimizer.zero_grad()
        return self.loss_sum.data.cpu().squeeze().numpy()[0]

    def step_loss(self, new_parameters, average_batch=True, keep_fine_losses=False):

        # Note: for the ACT step loss, we're only summing over the number of samples (dim1), for META model
        # we also sum over dim0 - the number of functions. But for ACT we need the losses per function in the
        # final_loss calculation (multiplied with the qt values, which we collect also for each function
        loss = get_step_loss(self.functions, new_parameters, avg_batch=False)
        if loss.dim() == 1:
            loss = loss.unsqueeze(1)
        # before we sum and optionally average, we keep the loss/function for the ACT loss computation later
        # Note, that the only difference with V1 is that here we append the Variable-loss, not the Tensor
        if keep_fine_losses:
            self.batch_step_losses.append(loss)
        if average_batch:
            return torch.mean(loss)
        else:
            return loss

    def compute_batch_loss(self, kl_scaler=1.):
        # construct the individual priors for each batch function, return DoubleTensor[batch_size, self.steps]
        g_priors = self.construct_priors()
        # truncate the self.qts variable to the maximum number of opt steps we made
        qts = self.qts[:, 0:self.step].double()
        # compute KL divergence term
        kl_term = kl_scaler * torch.mean(kl_divergence(qts, g_priors))
        # compute final loss, in which we multiply each loss by the qt time step values
        losses = torch.cat(self.batch_step_losses, 1).double()
        # sum over the time steps (dim 1) and average over the batch dimension
        losses = torch.mean(torch.sum(torch.mul(losses, qts), 1), 0)
        self.loss_sum = (losses + kl_term).squeeze()

    def set_qt_values(self, new_probs, next_iteration_condition, final_func_mask):
        qt = Variable(torch.zeros(new_probs.size(0)))
        next = torch.sum(next_iteration_condition).data.cpu().numpy()[0]
        finals = torch.sum(final_func_mask).data.cpu().squeeze().numpy()[0]
        next_idx = next_iteration_condition.data.squeeze().nonzero().squeeze()
        final_idx = final_func_mask.data.squeeze().nonzero().squeeze()
        if new_probs.is_cuda:
            qt = qt.cuda()
        if next > 0:
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
        g_priors = geom.pmf(R, self.geom_shape_param)
        # create truncated priors
        g_priors *= 1./np.sum(g_priors, 1).reshape((self.batch_size, 1))
        g_priors = Variable(torch.from_numpy(g_priors).double())
        if self.qts.is_cuda:
            g_priors = g_priors.cuda()
        return g_priors

    def process_step0(self, exper, loss):
        baseline_loss = torch.mean(loss, 0).data.cpu().squeeze().numpy()[0]
        exper.add_step_loss(baseline_loss, self.step)
        exper.add_opt_steps(self.step)







