import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict
from layer_lstm import LayerLSTMCell
from utils.config import config
from utils.regression import neg_log_likelihood_loss, nll_with_t_dist, RegressionFunction, RegressionWithStudentT


def generate_reversed_cum(q_probs_in, priors_in, use_cuda=False):
    T = q_probs_in.size(1)
    if use_cuda:
        q_probs = Variable(torch.DoubleTensor(q_probs_in.size()).cuda())
        priors = Variable(torch.DoubleTensor(priors_in.size()).cuda())
    else:
        q_probs = Variable(torch.DoubleTensor(q_probs_in.size()))
        priors = Variable(torch.DoubleTensor(priors_in.size()))
    q_probs[:, 0] = 1.
    priors[:, 0] = 1.
    for t in np.arange(1, T):
        q_probs[:, t] = 1. - torch.sum(q_probs_in[:, 0:t], 1)
        priors[:, t] = 1. - torch.sum(priors_in[:, 0:t], 1)

    return q_probs, priors


def kl_divergence(q_probs=None, prior_probs=None, threshold=-1e-4):
    """
    Kullback-Leibler divergence loss

    NOTE: summing over dimension 1, the optimization steps

    :param q_probs: the approximated posterior probability q(t|T). [batch_size, opt_steps]
    :param prior_probs: the prior probability p(t|T)
    :return: KL divergence loss
    """
    eps = 1e-90
    try:
        kl_div = torch.sum(q_probs * (torch.log(q_probs + eps) - torch.log(prior_probs + eps)), 1)
    except RuntimeError:
        print("q_probs.size ", q_probs.size())
        print("prior_probs.size ", prior_probs.size())
        raise RuntimeError("Running away from here...")
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise ValueError

    if np.any(kl_div.data.cpu().numpy() < 0):
        kl_np = kl_div.data.cpu().numpy()
        if np.any(kl_np[(kl_np < 0) & (kl_np < threshold)]):
            print("************* Negative KL-divergence *****************")
            print("sum(q_probs {:.2f}".format(torch.sum(q_probs.data.cpu().squeeze())))
            print("sum(prior_probs {:.2f}".format(torch.sum(prior_probs.data.cpu().squeeze())))
            kl_str = np.array_str(kl_div.data.cpu().squeeze().numpy(), precision=4)
            raise ValueError("KL divergence can't be less than zero {}".format(kl_str))

    return kl_div


def get_step_loss(optimizee_obj, new_parameters, avg_batch=False):
    if optimizee_obj.__class__ == RegressionFunction:
        loss = neg_log_likelihood_loss(optimizee_obj.y, optimizee_obj.y_t(new_parameters),
                                       stddev=optimizee_obj.stddev, N=optimizee_obj.n_samples,
                                       avg_batch=avg_batch, size_average=False)
    elif optimizee_obj.__class__ == RegressionWithStudentT:
        loss = nll_with_t_dist(optimizee_obj.y, optimizee_obj.y_t(new_parameters), N=optimizee_obj.n_samples,
                               shape_p=optimizee_obj.shape_p, scale_p=optimizee_obj.scale_p,
                               avg_batch=avg_batch)
    else:
        raise ValueError("Optimizee class not supported {}".format(optimizee_obj.__class__))

    return loss


def init_stat_vars(conf=None):
    if conf is None:
        T = config.T
        max_val_opt_steps = config.max_val_opt_steps
        conf = config
    else:
        T = conf.T
        max_val_opt_steps = conf.max_val_opt_steps

    # is used to compute the mean weights/probs over epochs
    qt_hist = OrderedDict([(i, np.zeros(i)) for i in np.arange(1, T + 1)])
    # the same object for the validation data
    qt_hist_val = OrderedDict([(i, np.zeros(i)) for i in np.arange(1, max_val_opt_steps + 1)])
    opt_step_hist = np.zeros(T)
    # the same object for the validation data
    opt_step_hist_val = np.zeros(max_val_opt_steps)
    # temporary variables to give some insights into the scale differences between log-likelihood terms and
    # kl-divergence terms
    ll_loss = np.zeros(conf.max_val_opt_steps)
    kl_div = np.zeros(conf.max_val_opt_steps)
    kl_entropy = np.zeros(conf.max_val_opt_steps)
    return qt_hist, qt_hist_val, opt_step_hist, opt_step_hist_val, ll_loss, kl_div, kl_entropy


class MetaLearner(nn.Module):

    def __init__(self, num_params_optimizee, num_inputs=1, num_hidden=20, num_layers=2, use_cuda=False,
                 output_bias=True, fg_bias=1):
        super(MetaLearner, self).__init__()
        self.name = "default"
        self.hidden_size = num_hidden
        self.use_cuda = use_cuda
        # number of parameters we need to optimize, for regression we choose the polynomial degree of the
        # regression function
        self.num_params = num_params_optimizee
        # self.linear1 = nn.Linear(2, num_hidden)
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.lstms = nn.ModuleList()
        for i in range(num_layers):
            self.lstms.append(LayerLSTMCell(num_hidden, num_hidden, forget_gate_bias=fg_bias))
            # self.lstms.append(nn.LSTMCell(num_hidden, num_hidden))

        self.linear_out = nn.Linear(num_hidden, 1, bias=output_bias)
        self.losses = []
        self.epochs_trained = 0
        if self.use_cuda:
            self.cuda()

    def cuda(self):
        super(MetaLearner, self).cuda()

    def zero_grad(self):
        super(MetaLearner, self).zero_grad()

    def reset_lstm(self, keep_states=False):

        if keep_states:
            for i in range(len(self.lstms)):
                self.hx[i] = Variable(self.hx[i].data)
                self.cx[i] = Variable(self.cx[i].data)
        else:
            self.hx = []
            self.cx = []
            # first loop over the number of parameters we have, because we're updating coordinate wise
            for i in range(len(self.lstms)):
                self.hx.append(Variable(torch.zeros(1, self.hidden_size)))
                self.cx.append(Variable(torch.zeros(1, self.hidden_size)))
                if self.use_cuda:
                    self.hx[i], self.cx[i] = self.hx[i].cuda(), self.cx[i].cuda()

    def forward(self, x_t):
        """
            x: contains the gradients of the loss function w.r.t. the regression function parameters
               Tensor shape: dim0=number of functions X dim1=number of parameters
            Coordinate wise processing:
        """
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(1)
        if self.use_cuda and not x_t.is_cuda:
            x_t = x_t.cuda()

        x_t = self.linear1(x_t)
        for i in range(len(self.lstms)):
            if x_t.size(0) != self.hx[i].size(0):
                self.hx[i] = self.hx[i].expand(x_t.size(0), self.hx[i].size(1))
                self.cx[i] = self.cx[i].expand(x_t.size(0), self.cx[i].size(1))

            self.hx[i], self.cx[i] = self.lstms[i](x_t, (self.hx[i], self.cx[i]))
            x_t = self.hx[i]
        x_t = self.linear_out(x_t)
        return x_t.squeeze()

    def meta_update(self, optimizee_with_grads, run_type="train"):

        """

        :rtype: object
        """
        # first determine whether the optimizee has base class nn.Module
        is_nn_module = nn.Module in optimizee_with_grads.__class__.__bases__
        if is_nn_module:
            delta_params = self(Variable(optimizee_with_grads.get_flat_grads().data))
        else:
            param_size = optimizee_with_grads.params.grad.size()
            flat_grads = Variable(optimizee_with_grads.params.grad.data.view(-1))
            # flat_params = Variable(optimizee_with_grads.params.data.view(-1).unsqueeze(1))
            # input = torch.cat((flat_grads, flat_params), 1)
            delta_params = self(flat_grads)
            # reshape parameters
            delta_params = delta_params.view(param_size)

        return delta_params

    def step_loss(self, optimizee_obj, new_parameters, average_batch=True):

        # looks odd that we don't pass "average_batch" parameter to get_stop_loss function, but we first compute
        # loss for all individual functions, then store the result and average (if necessary) afterwards
        loss = get_step_loss(optimizee_obj, new_parameters, avg_batch=False)
        # passing it as a new Variable breaks the backward...actually not necessary here, but for actV1 model
        if loss.dim() == 1:
            loss = loss.unsqueeze(1)
        self.losses.append(Variable(loss.data))
        if average_batch:
            return torch.mean(loss)
        else:
            return loss

    @property
    def sum_grads(self):
        sum_grads = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                sum_grads += torch.sum(param.grad.data)
            else:
                print("WARNING - No gradients!!!!")

        return sum_grads

    @property
    def sum_weights(self):
        weight_sum = 0
        for name, param in self.named_parameters():
            weight_sum += torch.sum(param.data)
        return weight_sum

    def reset_losses(self):
        self.losses = []

    def final_loss(self, loss_weights):
        losses = torch.mean(torch.cat(self.losses, 1), 0).squeeze()
        return torch.sum(losses * loss_weights)


class MetaLearnerWithValueFunction(MetaLearner):
    def __init__(self, num_params_optimizee, num_inputs=1, num_hidden=20, num_layers=2, use_cuda=False,
                 output_bias=True):
        super(MetaLearnerWithValueFunction, self).__init__(num_params_optimizee, num_inputs, num_hidden, num_layers,
                                                           use_cuda, output_bias=output_bias, fg_bias=1)

    def step_loss(self, optimizee_obj, new_parameters, average_batch=True):

        # looks odd that we don't pass "average_batch" parameter to get_stop_loss function, but we first compute
        # loss for all individual functions, then store the result and average (if necessary) afterwards
        loss = get_step_loss(optimizee_obj, new_parameters, avg_batch=False)
        # ACTUALLY THIS IS WHY I CLONED THE STEP_LOSS METHOD from MetaLearner, because I need to append the loss
        # without passing loss.data, in order to be able to execute .backward() later on
        if loss.dim() == 1:
            loss = loss.unsqueeze(1)
        self.losses.append(loss)
        if average_batch:
            return torch.mean(loss)
        else:
            return loss

    def final_loss_weighted(self, loss_weights, run_type="train"):
        T = len(self.losses)
        w = Variable(torch.arange(1, T+1))
        if self.use_cuda:
            w = w.cuda()
        self.losses = torch.cat(self.losses, 1)
        self.losses = torch.mul(self.losses, w.unsqueeze(0).expand_as(self.losses))
        self.losses = torch.mul(self.losses, loss_weights.unsqueeze(0).expand_as(self.losses))
        return torch.mean(torch.sum(self.losses, 1), 0).squeeze()

    def final_loss(self, loss_weights, run_type="train"):
        # make a matrix out of list, results in tensor [batch_size, num_of_time_steps]
        T = len(self.losses)
        self.losses = torch.cat(self.losses, 1)
        losses = Variable(torch.zeros(self.losses.size()))
        if self.use_cuda:
            losses = losses.cuda()
        # now we loop backwards through the time steps, we skip step T
        losses[:, T-1] = self.losses[:, T-1]
        for t in np.arange(2, T+1):
            l = self.losses[:, T-t:]
            w = loss_weights[0:t].unsqueeze(0).expand_as(l)
            losses[:, T-t] = torch.sum(l * w, 1)
        # if run_type != "train":
        #    print(loss_weights.data.cpu().squeeze().numpy())
        #    print(torch.mean(losses, 0).data.cpu().squeeze().numpy())
        # finally sum over all time steps for each function and then average over batch

        return torch.mean(torch.sum(losses, 1), 0).squeeze()


class MetaStepLearner(MetaLearner):
    """
        Implementation halve way. Should be a MetaLearner that also learns the loss-weights
        Current implementation learn those weights without restriction.
        Need to incorporate the constraint that they must sum to one
        Note: Can be started with --learner=meta --version=V4
    """

    def __init__(self, num_params_optimizee, num_inputs=1, num_hidden=20, num_layers=2, use_cuda=False,
                 output_bias=True):
        super(MetaStepLearner, self).__init__(num_params_optimizee, num_inputs, num_hidden, num_layers, use_cuda,
                                              output_bias=output_bias, fg_bias=1)
        self.q_t = []
        self.qt_linear_out = nn.Linear(num_hidden, 1, bias=True)
        if self.use_cuda:
            self.cuda()

    def forward(self, x_t):

        if self.use_cuda and not x_t.is_cuda:
            x_t = x_t.cuda()

        x_t = x_t.unsqueeze(1)
        x_t = self.linear1(x_t)
        for i in range(len(self.lstms)):
            if x_t.size(0) != self.hx[i].size(0):
                self.hx[i] = self.hx[i].expand(x_t.size(0), self.hx[i].size(1))
                self.cx[i] = self.cx[i].expand(x_t.size(0), self.cx[i].size(1))

            self.hx[i], self.cx[i] = self.lstms[i](x_t, (self.hx[i], self.cx[i]))
            x_t = self.hx[i]

        theta_out = self.linear_out(x_t)
        qt_out = self.qt_linear_out(x_t)
        # qt_out = self.act_qt_out (qt_out.view(param_shape))

        return tuple((theta_out.squeeze(), qt_out.squeeze()))

    def meta_update(self, optimizee_with_grads):

        """

        :rtype: object
        """
        # first determine whether the optimizee has base class nn.Module
        is_nn_module = nn.Module in optimizee_with_grads.__class__.__bases__
        if is_nn_module:
            param_size = list([optimizee_with_grads.num_of_funcs, optimizee_with_grads.dim])
            delta_theta, delta_qt = self(Variable(optimizee_with_grads.get_flat_params().data.view(-1)))
            delta_qt = torch.mean(delta_qt.view(*param_size), 1)
        else:
            param_size = optimizee_with_grads.params.grad.size()
            flat_grads = Variable(optimizee_with_grads.params.grad.data.view(-1))
            delta_theta, delta_qt = self(flat_grads)
            # reshape parameters
            delta_theta = delta_theta.view(param_size)

        return delta_theta, delta_qt

    def reset_losses(self):
        self.losses = []
        self.q_t = []


class AdaptiveMetaLearnerV2(MetaLearner):

    def __init__(self, num_params_optimizee, num_inputs=1, num_hidden=20, num_layers=2, use_cuda=False, output_bias=True):
        super(AdaptiveMetaLearnerV2, self).__init__(num_params_optimizee, num_inputs, num_hidden, num_layers, use_cuda,
                                                    output_bias=output_bias, fg_bias=1)

        # holds losses from each optimizer step
        self.losses = []
        self.q_t = []
        # number of parameters of the function to optimize (the optimizee)
        self.num_params = num_params_optimizee
        # currently we're only feeding the second LSTM with one parameter, the sum of the incoming gradients
        # of the optimizee
        self.act_num_params = 1
        self.qt_hist, self.qt_hist_val, self.opt_step_hist, self.opt_step_hist_val, self.ll_loss, self.kl_div, \
            self.kl_entropy = init_stat_vars()

        self.eps = 1e-90
        # temporary terms --- end
        self.q_soft = None
        # Parameters of the model
        # self.input_batch_norm = torch.nn.BatchNorm1d(self.num_params)
        self.linear_out = nn.Linear(num_hidden, 1, bias=output_bias)
        self.act_linear_out = nn.Linear(num_hidden, 1, bias=True)
        # self.act_qt_out = nn.Linear(self.num_params, 1, bias=output_bias)
        if self.use_cuda:
            self.cuda()

    def zero_grad(self):
        super(AdaptiveMetaLearnerV2, self).zero_grad()

    def reset_lstm(self, keep_states=False):
        super(AdaptiveMetaLearnerV2, self).reset_lstm(keep_states)

    def forward(self, x_t, param_shape=None):

        if self.use_cuda and not x_t.is_cuda:
            x_t = x_t.cuda()

        x_t = x_t.unsqueeze(1)
        x_t = self.linear1(x_t)
        for i in range(len(self.lstms)):
            if x_t.size(0) != self.hx[i].size(0):
                self.hx[i] = self.hx[i].expand(x_t.size(0), self.hx[i].size(1))
                self.cx[i] = self.cx[i].expand(x_t.size(0), self.cx[i].size(1))

            self.hx[i], self.cx[i] = self.lstms[i](x_t, (self.hx[i], self.cx[i]))
            x_t = self.hx[i]

        theta_out = self.linear_out(x_t)
        qt_out = self.act_linear_out(x_t)
        # qt_out = self.act_qt_out (qt_out.view(param_shape))

        return tuple((theta_out.squeeze(), qt_out.squeeze()))

    def meta_update(self, optimizee_with_grads):
        """

            :param optimizee_with_grads:
            :return: delta theta, delta qt
        """

        param_size = optimizee_with_grads.params.grad.size()

        flat_grads = Variable(optimizee_with_grads.params.grad.data.view(-1))
        delta_theta, delta_qt = self(flat_grads, param_size)
        # reshape parameters
        delta_theta = delta_theta.view(param_size)
        # reshape qt-values and calculate mean
        delta_qt = torch.mean(delta_qt.view(param_size), 1).unsqueeze(1)
        return delta_theta, delta_qt

    def step_loss(self, optimizee_obj, new_parameters, average_batch=True):

        # Note: for the ACT step loss, we're only summing over the number of samples (dim1), for META model
        # we also sum over dim0 - the number of functions. But for ACT we need the losses per function in the
        # final_loss calculation (multiplied with the qt values, which we collect also for each function
        loss = get_step_loss(optimizee_obj, new_parameters, avg_batch=False)

        # before we sum and optionally average, we keep the loss/function for the ACT loss computation later
        # Note, that the only difference with V1 is that here we append the Variable-loss, not the Tensor
        if loss.dim() == 1:
            loss = loss.unsqueeze(1)
        self.losses.append(loss)
        if average_batch:
            return torch.mean(loss)
        else:
            return loss

    def final_lossV2(self, prior_probs, run_type='train'):
        """
            Calculates the loss of the ACT model.
            converts lists self.losses & self.q_t which contain the q(t|T) and loss_t values for each function
                over the T time-steps.
            The final dimensions of losses & q_t is [batch_size, opt_steps]
            KL-divergence is computed for each batch function. For the final loss value the D_KLs are averaged

        :param prior_probs: the prior probabilities of the stopping-distribution p(t|T).
                            has dimensions: [batch_size, opt_steps]
        :param run_type: train/valid, indicates in which dictionary we need to save the result for later analysis
        :return: scalar loss
        """
        assert len(self.losses) == len(self.q_t), "Length of two objects is not the same!"
        eps = 1e-10
        self.q_soft = None
        # number of steps is in dimension 1 of prior probabilities
        num_of_steps = prior_probs.size(1)
        rl_weights = Variable(torch.arange(1, num_of_steps+1).double()).unsqueeze(0)
        if self.use_cuda:
            rl_weights = rl_weights.cuda()

        # concatenate everything along dimension 1, the number of time-steps
        losses = torch.cat(self.losses, 1)
        q_t = torch.cat(self.q_t, 1)
        self.q_soft = F.softmax(q_t.double())
        q, p = generate_reversed_cum(self.q_soft, prior_probs, self.use_cuda)
        rl_weights = rl_weights.expand_as(self.q_soft)
        if np.isnan(np.sum(q.data.cpu().numpy())):
            print("**** Nan value in q tensor ****")
        kl_loss = torch.mul(rl_weights , q * (torch.log(q + eps) - torch.log(p + eps)))
        if np.isnan(np.sum(kl_loss.data.cpu().numpy())):
            print("**** Nan value in kl_loss ****")
        losses = torch.mul(rl_weights, losses.double())
        # print(kl_loss[10, :].data.cpu().squeeze().numpy())
        kl_loss = torch.sum(kl_loss, 1)
        # print(torch.mul(q, losses)[10, :].data.cpu().squeeze().numpy())
        loss = (torch.mean(torch.sum(torch.mul(q, losses), 1) + kl_loss, 0)).squeeze()
        qts = torch.mean(self.q_soft, 0).data.cpu().squeeze().numpy()
        if run_type == 'train':
            self.qt_hist[num_of_steps] += qts
            self.opt_step_hist[num_of_steps - 1] += 1
        elif run_type == 'val':
            self.ll_loss += torch.mean(losses.double(), 0).data.cpu().squeeze().numpy()
            self.kl_div += torch.mean(torch.log(prior_probs.double() + self.eps), 0).data.cpu().squeeze().numpy()
            self.kl_entropy += torch.mean(torch.log(self.q_soft + self.eps), 0).data.cpu().squeeze().numpy()

        return loss

    def final_loss(self, prior_probs, run_type='train'):
        """
            Calculates the loss of the ACT model.
            converts lists self.losses & self.q_t which contain the q(t|T) and loss_t values for each function
                over the T time-steps.
            The final dimensions of losses & q_t is [batch_size, opt_steps]
            KL-divergence is computed for each batch function. For the final loss value the D_KLs are averaged

        :param prior_probs: the prior probabilities of the stopping-distribution p(t|T).
                            has dimensions: [batch_size, opt_steps]
        :param run_type: train/valid, indicates in which dictionary we need to save the result for later analysis
        :return: scalar loss
        """
        assert len(self.losses) == len(self.q_t), "Length of two objects is not the same!"
        self.q_soft = None
        # number of steps is in dimension 1 of prior probabilities
        num_of_steps = prior_probs.size(1)
        # concatenate everything along dimension 1, the number of time-steps
        losses = torch.cat(self.losses, 1)
        q_t = torch.cat(self.q_t, 1)
        self.q_soft = F.softmax(q_t.double())
        kl_loss = kl_divergence(q_probs=self.q_soft, prior_probs=prior_probs.double())
        loss = (torch.mean(torch.sum(self.q_soft * losses.double(), 1) + kl_loss, 0)).squeeze()
        qts = torch.mean(self.q_soft, 0).data.cpu().squeeze().numpy()
        if run_type == 'train':
            self.qt_hist[num_of_steps] += qts
            self.opt_step_hist[num_of_steps - 1] += 1
        elif run_type == 'val':
            self.ll_loss += torch.mean(losses.double(), 0).data.cpu().squeeze().numpy()
            self.kl_div += torch.mean(torch.log(prior_probs.double() + self.eps), 0).data.cpu().squeeze().numpy()
            self.kl_entropy += torch.mean(torch.log(self.q_soft + self.eps), 0).data.cpu().squeeze().numpy()

        return loss

    def reset_final_loss(self):
        self.losses = []
        self.q_t = []
        self.q_soft = None

    def init_qt_statistics(self, conf=None):
        self.qt_hist, self.qt_hist_val, self.opt_step_hist, self.opt_step_hist_val, self.ll_loss, self.kl_div, \
            self.kl_entropy = init_stat_vars(conf)


class AdaptiveMetaLearnerV1(AdaptiveMetaLearnerV2):
    def __init__(self, num_params_optimizee, num_inputs=1, num_hidden=20, num_layers=2,
                 use_cuda=False, output_bias=True):
        super(AdaptiveMetaLearnerV1, self).__init__(num_params_optimizee, num_inputs, num_hidden, num_layers, use_cuda,
                                                    output_bias=output_bias)

        self.num_hidden_act = num_hidden
        # holds losses from each optimizer step
        self.act_num_params = self.num_params
        # this is the LSTM for the ACT distribution
        self.act_linear1 = nn.Linear(num_inputs, self.num_hidden_act)
        # self.act_ln1 = LayerNorm1D(self.num_hidden_act)
        self.act_lstms = nn.ModuleList()
        for i in range(num_layers):
            self.act_lstms.append(LayerLSTMCell(self.num_hidden_act, self.num_hidden_act, forget_gate_bias=1.))

        self.act_linear_out = nn.Linear(self.num_hidden_act, 1, bias=output_bias)

    def reset_lstm(self, keep_states=False):
        super(AdaptiveMetaLearnerV1, self).reset_lstm(keep_states)

        if keep_states:
            for i in range(len(self.act_lstms)):
                self.act_hx[i] = Variable(self.act_hx[i].data)
                self.act_cx[i] = Variable(self.act_cx[i].data)
        else:
            self.act_hx = []
            self.act_cx = []
            # first loop over the number of parameters we have, because we're updating coordinate wise
            for i in range(len(self.act_lstms)):
                self.act_hx.append(Variable(torch.zeros(1, self.hidden_size)))
                self.act_cx.append(Variable(torch.zeros(1, self.hidden_size)))
                if self.use_cuda:
                    self.act_hx[i], self.act_cx[i] = self.act_hx[i].cuda(), self.act_cx[i].cuda()

    def forward(self, x):
        if self.use_cuda and not x.is_cuda:
            x = x.cuda()

        x = x.unsqueeze(1)
        x_t = self.linear1(x)
        # act input
        q_t = self.act_linear1(x)
        # assuming that both LSTM (for L2L and ACT) have same number of layers.
        for i in range(len(self.lstms)):
            if x_t.size(0) != self.hx[i].size(0):
                self.hx[i] = self.hx[i].expand(x_t.size(0), self.hx[i].size(1))
                self.cx[i] = self.cx[i].expand(x_t.size(0), self.cx[i].size(1))
            # act lstm part
            if q_t.size(0) != self.act_hx[i].size(0):
                self.act_hx[i] = self.act_hx[i].expand(q_t.size(0), self.act_hx[i].size(1))
                self.act_cx[i] = self.act_cx[i].expand(q_t.size(0), self.act_cx[i].size(1))

            self.hx[i], self.cx[i] = self.lstms[i](x_t, (self.hx[i], self.cx[i]))
            # act lstm part
            self.act_hx[i], self.act_cx[i] = self.act_lstms[i](q_t, (self.act_hx[i], self.act_cx[i]))
            x_t = self.hx[i]
            q_t = self.act_hx[i]

        theta_out = self.linear_out(x_t)
        qt_out = self.act_linear_out(q_t)

        return tuple((theta_out.squeeze(), qt_out.squeeze()))

    def step_loss(self, optimizee_obj, new_parameters, average_batch=True):
        N = float(optimizee_obj.n_samples)
        # Note: for the ACT step loss, we're only summing over the number of samples (dim1), for META model
        # we also sum over dim0 - the number of functions. But for ACT we need the losses per function in the
        # final_loss calculation (multiplied with the qt values, which we collect also for each function
        loss = neg_log_likelihood_loss(optimizee_obj.y, optimizee_obj.y_t(params=new_parameters),
                                       stddev=optimizee_obj.stddev, N=N, size_average=False)

        # before we sum and optionally average, we keep the loss/function for the ACT loss computation later
        # Note, that the only difference with V1 is that here we append the Variable-loss, not the Tensor
        self.losses.append(Variable(loss.data))
        if average_batch:
            return torch.mean(loss)
        else:
            return loss
