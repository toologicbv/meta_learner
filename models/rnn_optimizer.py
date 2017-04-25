import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from collections import OrderedDict

from layer_norm_lstm import LayerNormLSTMCell
from utils.config import config


def kl_divergence(q_probs=None, prior_probs=None, do_average=True):
    """
    Kullback-Leibler divergence loss

    :param q_probs: the approximated posterior probability q(t|T)
    :param prior_probs: the prior probability p(t|T)
    :return: KL divergence loss
    """
    try:
        kl_div = torch.sum(q_probs * (torch.log(q_probs) - torch.log(prior_probs)))
    except RuntimeError:
        print("q_probs.size ", q_probs.size())
        print("prior_probs.size ", prior_probs.size())
        raise RuntimeError("Running away from here...")
    except:
        print("Unexpected error:", sys.exc_info()[0])
        raise ValueError

    if kl_div.data.squeeze().numpy()[0] < 0:
        print("************* Negative KL-divergence *****************")
        print("sum(q_probs {:.2f}".format(torch.sum(q_probs.data.squeeze())))
        print("sum(prior_probs {:.2f}".format(torch.sum(prior_probs.data.squeeze())))
        raise ValueError("KL divergence can't be less than zero {:.3.f}".format(kl_div.data.squeeze().numpy()[0]))
    if do_average:
        n = Variable(torch.FloatTensor([q_probs.size(1)]))
        return 1/n * kl_div
    else:
        return kl_div


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
        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.lstms = nn.ModuleList()
        for i in range(num_layers):
            self.lstms.append(LayerNormLSTMCell(num_hidden, num_hidden, forget_gate_bias=fg_bias))

        self.linear_out = nn.Linear(num_hidden, 1, bias=output_bias)
        self.losses = []

    def cuda(self):
        super(MetaLearner, self).cuda()
        for i in range(len(self.lstms)):
            self.lstms[i].cuda()

    def zero_grad(self):
        super(MetaLearner, self).zero_grad()

    def reset_lstm(self, keep_states=False):

        if keep_states:
            for theta in np.arange(self.num_params):
                for i in range(len(self.lstms)):
                    self.hx[theta][i] = Variable(self.hx[theta][i].data)
                    self.cx[theta][i] = Variable(self.cx[theta][i].data)
        else:
            self.hx = {}
            self.cx = {}
            # first loop over the number of parameters we have, because we're updating coordinate wise
            for theta in np.arange(self.num_params):
                self.hx[theta] = [Variable(torch.zeros(1, self.hidden_size)) for i in range(len(self.lstms))]
                self.cx[theta] = [Variable(torch.zeros(1, self.hidden_size)) for i in range(len(self.lstms))]

                if self.use_cuda:
                    for i in range(len(self.lstms)):
                        self.hx[theta][i], self.cx[theta][i] = self.hx[theta][i].cuda(), self.cx[theta][i].cuda()

    def forward(self, x):
        """
            x: contains the gradients of the loss function w.r.t. the regression function parameters
               Tensor shape: dim0=number of functions X dim1=number of parameters
            Coordinate wise processing:
        """
        if self.use_cuda:
            x = x.cuda()

        x_out = []
        for t in np.arange(self.num_params):
            x_t = x[:, t].unsqueeze(1)
            x_t = self.linear1(x_t)
            for i in range(len(self.lstms)):
                if x_t.size(0) != self.hx[t][i].size(0):
                    self.hx[t][i] = self.hx[t][i].expand(x_t.size(0), self.hx[t][i].size(1))
                    self.cx[t][i] = self.cx[t][i].expand(x_t.size(0), self.cx[t][i].size(1))

                self.hx[t][i], self.cx[t][i] = self.lstms[i](x_t, (self.hx[t][i], self.cx[t][i]))
                x_t = self.hx[t][i]
            x_t = self.linear_out(x_t)
            x_out.append(x_t)

        res = torch.cat(x_out, 1)
        return res

    def meta_update(self, func_with_grads):

        """

        :rtype: object
        """
        grads = Variable(func_with_grads.params.grad.data)
        delta_params = self(grads)

        return delta_params

    def step_loss(self, reg_func_obj, new_parameters, average=True):

        if average:
            N = float(reg_func_obj.num_of_funcs)
        else:
            N = 1.
        loss = 1/N * 0.5 * 1/reg_func_obj.noise_sigma * \
               torch.sum((reg_func_obj.y - reg_func_obj.get_y_values(new_parameters)) ** 2)
        self.losses.append(Variable(loss.data))
        return loss

    @property
    def sum_grads(self):
        sum_grads = 0
        for i, cells in enumerate(self.lstms):
            for j, module in enumerate(cells.children()):
                for param_name, params in module._parameters.iteritems():
                    if params.grad is not None:
                        sum_grads += torch.sum(params.grad.data)
        # for param in self.parameters():
        #    sum_grads += torch.sum(param.grad.data)

        return sum_grads

    def reset_losses(self):
        self.losses = []


class AdaptiveMetaLearnerV1(MetaLearner):
    def __init__(self, num_params_optimizee, num_inputs=1, num_hidden=20, num_layers=2, use_cuda=False, output_bias=True):
        super(AdaptiveMetaLearnerV1, self).__init__(num_params_optimizee, num_inputs, num_hidden, num_layers, use_cuda,
                                                    output_bias=output_bias, fg_bias=1.)
        self.linear_grads = nn.Linear(self.num_params, 1)
        self.num_hidden_act = num_hidden
        # holds losses from each optimizer step
        self.losses = []
        self.q_t = []
        # number of parameters of the function to optimize (the optimizee)
        self.num_params = self.opt_wrapper.optimizee.parameter.data.view(-1).size(0)
        # currently we're only feeding the second LSTM with one parameter, the sum of the incoming gradients
        # of the optimizee
        self.act_num_params = self.num_params
        # is used to compute the mean weights/probs over epochs
        self.qt_hist = OrderedDict([(i, np.zeros(i)) for i in np.arange(1, config.T + 1)])
        # the same object for the validation data
        self.qt_hist_val = OrderedDict([(i, np.zeros(i)) for i in np.arange(1, config.max_val_opt_steps + 1)])
        self.q_soft = None
        self.opt_step_hist = np.zeros(config.T)
        # the same object for the validation data
        self.opt_step_hist_val = np.zeros(config.max_val_opt_steps)
        # this is the LSTM for the ACT distribution
        self.act_linear1 = nn.Linear(num_inputs, self.num_hidden_act)
        # self.act_ln1 = LayerNorm1D(self.num_hidden_act)
        self.act_lstms = nn.ModuleList()
        for i in range(num_layers):
            self.act_lstms.append(LayerNormLSTMCell(self.num_hidden_act, self.num_hidden_act, forget_gate_bias=1.))

        self.lambda_q = nn.Parameter(torch.zeros(1, 1))
        torch.nn.init.uniform(self.lambda_q, -0.1, 0.1)
        self.act_linear_out = nn.Linear(self.num_hidden_act, 1, bias=output_bias)

    def zero_grad(self):
        super(AdaptiveMetaLearnerV1, self).zero_grad()
        # make sure we also reset the gradients of the LSTM cells as well
        for i, cells in enumerate(self.act_lstms):
            cells.zero_grad()
        self.act_linear1.zero_grad()
        # self.act_ln1.zero_grad()
        self.act_linear_out.zero_grad()

    def reset_lstm(self, func, keep_states=False):
        super(AdaptiveMetaLearnerV1, self).reset_lstm(func, keep_states)

        if keep_states:
            for theta in np.arange(self.act_num_params):
                for i in range(len(self.act_lstms)):
                    self.act_hx[theta][i] = Variable(self.act_hx[theta][i].data)
                    self.act_cx[theta][i] = Variable(self.act_cx[theta][i].data)
        else:
            self.act_hx = {}
            self.act_cx = {}
            # first loop over the number of parameters we have, because we're updating coordinate wise
            for theta in np.arange(self.act_num_params):
                self.act_hx[theta] = [Variable(torch.zeros(1, self.num_hidden_act)) for i in range(len(self.act_lstms))]
                self.act_cx[theta] = [Variable(torch.zeros(1, self.num_hidden_act)) for i in range(len(self.act_lstms))]

                if self.use_cuda:
                    for i in range(len(self.act_lstms)):
                        self.act_hx[theta][i], self.act_cx[theta][i] = self.act_hx[theta][i].cuda(), \
                                                                       self.act_cx[theta][i].cuda()

    def forward(self, x):
        if self.use_cuda:
            x = x.cuda()

        x_out = []
        qt_out = []
        for t in np.arange(self.num_params):
            x_t = x[t].unsqueeze(1)
            q_t = x[t].unsqueeze(1)
            x_t = self.linear1(x_t)
            # act input
            q_t = self.act_linear1(q_t)
            # assuming that both LSTM (for L2L and ACT) have some number of layers.
            for i in range(len(self.lstms)):
                if x_t.size(0) != self.hx[t][i].size(0):
                    self.hx[t][i] = self.hx[t][i].expand(x_t.size(0), self.hx[t][i].size(1))
                    self.cx[t][i] = self.cx[t][i].expand(x_t.size(0), self.cx[t][i].size(1))
                # act lstm part
                if q_t.size(0) != self.act_hx[t][i].size(0):
                    self.act_hx[t][i] = self.act_hx[t][i].expand(q_t.size(0), self.act_hx[t][i].size(1))
                    self.act_cx[t][i] = self.act_cx[t][i].expand(q_t.size(0), self.act_cx[t][i].size(1))

                self.hx[t][i], self.cx[t][i] = self.lstms[i](x_t, (self.hx[t][i], self.cx[t][i]))
                # act lstm part
                self.act_hx[t][i], self.act_cx[t][i] = self.act_lstms[i](q_t, (self.act_hx[t][i], self.act_cx[t][i]))

                x_t = self.hx[t][i]
                q_t = self.act_hx[t][i]

            x_t = self.linear_out(x_t)
            q_t = self.lambda_q * F.tanh(self.act_linear_out(q_t))
            x_out.append(x_t)
            qt_out.append(q_t)
        x_out = torch.cat(x_out)
        qt_out = torch.mean(torch.cat(qt_out), 0)

        return tuple((x_out, qt_out))

    def meta_update(self, func_with_grads):
        """
            We use the quadratic loss surface - which is a copy of the "outside" loss surface that we're trying to optimize
            - to (a) compute the new parameters of this loss surface by taking the gradients of the outside loss surface
            pass them through our LSTM and subtract (gradient descent) the LSTM output from the "inside" loss surface
            parameters.
            (b) we copy the new parameter of the "inside" loss surface
            :param func_with_grads:
            :return:
        """

        grads = Variable(func_with_grads.parameter.grad.data)
        delta_grads, qt = self(grads)
        # collect the parts we need to calculate the final loss over all time-steps

        return delta_grads, qt

    def step_loss(self, new_parameters):
        # copy updated parameters to our "inside" loss surface model
        self.opt_wrapper.set_parameters(new_parameters)
        loss = torch.sum(self.opt_wrapper.optimizee.func(new_parameters))
        self.losses.append(Variable(loss.data))
        return loss

    def final_loss(self, prior_probs, run_type='train'):
        assert len(self.losses) == len(self.q_t), "Length of two objects is not the same!"
        self.q_soft = None
        num_of_steps = prior_probs.size(0)
        self.losses = torch.cat(self.losses, 0)
        self.q_t = torch.cat(self.q_t, 1)
        self.q_soft = F.softmax(self.q_t)
        # concatenate all qt factors and calculate probabilities, NOTE, q_t tensor has size (1, opt-steps)
        # and therefore we compute softmax over dimension 1
        # Note, because we are computing the KL-divergence as "sum q(t|T) * (log q(t|T) - log p(t|T))"
        # we are adding the KL divergence to the loss and not subtracting it.
        loss = torch.mean(self.q_soft * self.losses) + kl_divergence(q_probs=self.q_soft, prior_probs=prior_probs,
                                                                     do_average=False)
        # loss1 = torch.mean(self.q_soft * losses)
        # loss2 = kl_divergence(self.q_soft, prior_probs, do_average=True)
        # print("loss+kl {:.4f} + {:.4f}".format(loss1.data.squeeze().numpy()[0],
        #                                       loss2.data.squeeze().numpy()[0]))
        # aggregate the probs so we can compute some average later...for debugging purposes
        # increase counter for this "number of optimization steps". we use this later to evaluate the
        # meta optimizer (note substract 1 because this is an array
        if run_type == 'train':
            self.qt_hist[num_of_steps] += self.q_soft.data.squeeze().numpy()
            self.opt_step_hist[num_of_steps - 1] += 1
        elif run_type == 'val':
            self.qt_hist_val[num_of_steps] += self.q_soft.data.squeeze().numpy()
            self.opt_step_hist_val[num_of_steps - 1] += 1

        return loss

    def reset_final_loss(self):
        self.losses = []
        self.q_t = []
        self.q_soft = None


class AdaptiveMetaLearnerV2(MetaLearner):

    def __init__(self, num_params_optimizee, num_inputs=1, num_hidden=20, num_layers=2, use_cuda=False, output_bias=True):
        super(AdaptiveMetaLearnerV2, self).__init__(num_params_optimizee, num_inputs, num_hidden, num_layers, use_cuda,
                                                    output_bias=output_bias, fg_bias=1)
        self.linear_grads = nn.Linear(self.num_params, 1)
        # holds losses from each optimizer step
        self.losses = []
        self.q_t = []
        # number of parameters of the function to optimize (the optimizee)
        self.num_params = num_params_optimizee
        # currently we're only feeding the second LSTM with one parameter, the sum of the incoming gradients
        # of the optimizee
        self.act_num_params = 1
        # is used to compute the mean weights/probs over epochs
        self.qt_hist = OrderedDict([(i, np.zeros(i)) for i in np.arange(1, config.T + 1)])
        # the same object for the validation data
        self.qt_hist_val = OrderedDict([(i, np.zeros(i)) for i in np.arange(1, config.max_val_opt_steps + 1)])
        self.q_soft = None
        self.opt_step_hist = np.zeros(config.T)
        # the same object for the validation data
        self.opt_step_hist_val = np.zeros(config.max_val_opt_steps)
        # separate affine layer for the q(t) factor
        self.lambda_q = nn.Parameter(torch.zeros(1, 1))
        torch.nn.init.uniform(self.lambda_q, -0.1, 0.1)
        self.linear_out = nn.Linear(num_hidden // 2, 1, bias=output_bias)
        self.act_linear_out = nn.Linear(num_hidden // 2, 1, bias=output_bias)

    def zero_grad(self):
        super(AdaptiveMetaLearnerV2, self).zero_grad()
        # make sure we also reset the gradients of the LSTM cells as well
        self.act_linear_out.zero_grad()

    def reset_lstm(self, func, keep_states=False):
        super(AdaptiveMetaLearnerV2, self).reset_lstm(func, keep_states)

    def forward(self, x):
        if self.use_cuda:
            x = x.cuda()

        x_out = []
        qt_out = []
        for t in np.arange(self.num_params):
            x_t = x[t].unsqueeze(1)
            x_t = self.linear1(x_t)
            for i in range(len(self.lstms)):
                if x_t.size(0) != self.hx[t][i].size(0):
                    self.hx[t][i] = self.hx[t][i].expand(x_t.size(0), self.hx[t][i].size(1))
                    self.cx[t][i] = self.cx[t][i].expand(x_t.size(0), self.cx[t][i].size(1))

                self.hx[t][i], self.cx[t][i] = self.lstms[i](x_t, (self.hx[t][i], self.cx[t][i]))
                x_t = self.hx[t][i]

            x_t, qt = x_t.split(self.hidden_size // 2, 1)
            x_out.append(self.linear_out(x_t))
            qt_out.append(self.lambda_q * F.tanh(self.act_linear_out(qt)))

        x_out = torch.cat(x_out)
        qt_out = torch.mean(torch.cat(qt_out), 0)

        return tuple((x_out, qt_out))

    def meta_update(self, func_with_grads):
        """
            We use the quadratic loss surface - which is a copy of the "outside" loss surface that we're trying to optimize
            - to (a) compute the new parameters of this loss surface by taking the gradients of the outside loss surface
            pass them through our LSTM and subtract (gradient descent) the LSTM output from the "inside" loss surface
            parameters.
            (b) we copy the new parameter of the "inside" loss surface
            :param func_with_grads:
            :return:
        """
        grads = Variable(func_with_grads.parameter.grad.data)
        delta_grads, delta_qt = self(grads)

        return delta_grads, delta_qt

    def step_loss(self, new_parameters):
        # copy updated parameters to our "inside" loss surface model
        self.opt_wrapper.set_parameters(new_parameters)
        loss = torch.sum(self.opt_wrapper.optimizee.func(new_parameters))
        # Note, here we append the autograd.Variable(loss) which will (in comparison to V1) be part
        # of the final back-prop because we only execute .backward() once per function (later per mini-batch).
        self.losses.append(loss)
        return loss

    def final_loss(self, prior_probs, run_type='train'):
        assert len(self.losses) == len(self.q_t), "Length of two objects is not the same!"
        self.q_soft = None
        num_of_steps = prior_probs.size(0)
        self.losses = torch.cat(self.losses, 0)
        self.q_t = torch.cat(self.q_t, 1)
        self.q_soft = F.softmax(self.q_t)
        # concatenate all qt factors and calculate probabilities, NOTE, q_t tensor has size (1, opt-steps)
        # and therefore we compute softmax over dimension 1
        # Note, because we are computing the KL-divergence as "sum q(t|T) * (log q(t|T) - log p(t|T))"
        # we are adding the KL divergence to the loss and not subtracting it.
        # NOTE: we SHOULD actually SUM the first terms...but for the 2D quadratics these values would be
        # much too huge in comparison to the KL terms because of the f(theta) terms in the loss (or step losses)
        # I tried it with sums...but the q(t) distribution stays flat then
        loss = torch.mean(self.q_soft * self.losses) + kl_divergence(q_probs=self.q_soft, prior_probs=prior_probs,
                                                                     do_average=False)
        if np.isnan(loss.data.numpy()):
            print(self.q_soft.data.numpy())
            print(self.losses.data.numpy())
        # increase counter for this "number of optimization steps". we use this later to evaluate the
        # meta optimizer (note subtract 1 because this is an array
        if run_type == 'train':
            self.qt_hist[num_of_steps] += self.q_soft.data.squeeze().numpy()
            self.opt_step_hist[num_of_steps - 1] += 1
        elif run_type == 'val':
            self.qt_hist_val[num_of_steps] += self.q_soft.data.squeeze().numpy()
            self.opt_step_hist_val[num_of_steps - 1] += 1

        return loss

    def reset_final_loss(self):
        self.losses = []
        self.q_t = []
        self.q_soft = None