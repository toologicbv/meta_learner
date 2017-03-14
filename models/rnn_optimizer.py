from functools import reduce
from operator import mul

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layer_norm_lstm import LayerNormLSTMCell
from layer_norm import LayerNorm1D


class MetaLearner(nn.Module):

    def __init__(self, func, num_inputs=1, num_hidden=20, bias=True, num_layers=2, use_cuda=False):
        super(MetaLearner, self).__init__()
        self.helper_func = func
        self.hidden_size = num_hidden
        self.use_cuda = use_cuda

        self.lstms = []
        for i in range(num_layers):
            if i == 0:
                # first layer, input to hidden dim
                self.lstms.append(LayerNormLSTMCell(num_inputs, num_hidden))
                # self.lstms.append(nn.LSTMCell(num_inputs, num_hidden))
            else:
                # all other layers have hidden to hidden size
                self.lstms.append(LayerNormLSTMCell(num_hidden, num_hidden))
                # self.lstms.append(nn.LSTMCell(num_hidden, num_hidden))

        self.linear_out = nn.Linear(num_hidden, 1)

        # Linear transformation to generate pi(t=k|grad x, T)
        self.linear_pi = nn.Linear(num_hidden, 1, bias=bias)
        self.linear_future = nn.Linear(num_hidden, 1, bias=bias)
        # variable used to
        self.loss_meta_model = None
        # array that collects loss components that we need for the final update of the T-optimizer
        self.q_t = None
        self.grads = Variable(torch.zeros(1), requires_grad=True)

    def cuda(self):
        super(MetaLearner, self).cuda()
        for i in range(len(self.lstms)):
            self.lstms[i].cuda()

    def reset_lstm(self, func, keep_states=False):
        # copy the quadratic function "func" to our internal quadratic function we keep in the meta optimizer
        self.helper_func.reset(func)
        self.loss_meta_model = None
        self.q_t = None
        if keep_states:
            for i in range(len(self.lstms)):
                self.hx[i] = Variable(self.hx[i].data)
                self.cx[i] = Variable(self.cx[i].data)
        else:
            self.hx = []
            self.cx = []
            for i in range(len(self.lstms)):
                self.hx.append(Variable(torch.zeros(1, self.hidden_size)))
                self.cx.append(Variable(torch.zeros(1, self.hidden_size)))
                if self.use_cuda:
                    self.hx[i], self.cx[i] = self.hx[i].cuda(), self.cx[i].cuda()

    def forward(self, x):
        # NO gradient pre-processing so far!

        for i in range(len(self.lstms)):
            if x.size(0) != self.hx[i].size(0):
                self.hx[i] = self.hx[i].expand(x.size(0), self.hx[i].size(1))
                self.cx[i] = self.cx[i].expand(x.size(0), self.cx[i].size(1))
            # print("xsize / self.hx[i].size / self.cx[i].size", x.size(), self.hx[i].size(), self.cx[i].size())
            self.hx[i], self.cx[i] = self.lstms[i](x, (self.hx[i], self.cx[i]))
            x = self.hx[i]

        # pi_t = F.sigmoid(self.linear_pi(x))
        # save the pi-components we need later to calc the softmax over all pi's from time steps
        # if self.q_t is None:
        #     self.q_t = pi_t
        #     self.loss_meta_model = loss_quad
        # else:
        #     self.q_t = torch.cat((self.q_t, pi_t), 1)
        #     self.loss_meta_model = torch.cat((self.loss_meta_model, loss_quad), 1)
        #
        # fut_t = F.sigmoid(self.linear_future(x))
        # let's start with the original paper and only output g(theta_t) produced by the RNN optimizer
        x_out = self.linear_out(x)
        return x_out.squeeze()

    def meta_update(self, func_with_grads):
        parameters = self.helper_func.helper_q2_func.parameter
        # print("1 - self.helper_func.helper_q2_func.parameter ", self.helper_func.helper_q2_func.parameter.requires_grad)
        # print("1 - parameters ", parameters.requires_grad)
        grads = Variable(func_with_grads.parameter.grad.data, requires_grad=True)
        # if self.use_cuda:
        #   grads.data.cuda()
        # print("grads.volatile ", grads.volatile, grads.requires_grad)
        parameters = parameters + self(grads)
        # print("2 - self.helper_func.helper_q2_func.parameter ", self.helper_func.helper_q2_func.parameter.requires_grad)
        # print("2 - parameters ", parameters.requires_grad)
        self.helper_func.set_parameters(parameters)
        # print("3 - self.helper_func.helper_q2_func.parameter ", self.helper_func.helper_q2_func.parameter.requires_grad)
        # copy new parameters also to "outside" func that deliverd the grads in the first place
        self.helper_func.copy_params_to(func_with_grads)

        return self.helper_func.helper_q2_func

    def loss_func(self):
        assert self.loss_meta_model.size()[0] == self.q_t.size()[0], "Length of two objects is not the same!"
        q_soft = F.softmax(self.q_t)
        # kl =
        return torch.mean(q_soft.mul(self.loss_meta_model))


class HelperQuadratic(object):
    """
        this class holds the quadratic function that we want to minimize by means of the meta-optimizer
        we need to reset
    """
    def __init__(self, func):
        self.helper_q2_func = func

    def reset(self, q_func):
        self.helper_q2_func.parameter = Variable(q_func.parameter.data, requires_grad=True)
        self.helper_q2_func.x_min = q_func.x_min
        self.helper_q2_func.x_max = q_func.x_max
        self.helper_q2_func.func = q_func.f
        self.helper_q2_func.true_opt = q_func.true_opt

    def set_parameters(self, parameters):
        self.helper_q2_func.parameter = parameters

    def copy_params_to(self, func):

        func.parameter = Variable(self.helper_q2_func.parameter.data, requires_grad=True)
