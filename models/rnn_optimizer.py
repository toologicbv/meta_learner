
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from layer_norm_lstm import LayerNormLSTMCell
from layer_norm import LayerNorm1D


class MetaLearner(nn.Module):

    def __init__(self, func, num_inputs=1, num_hidden=20, bias=True, num_layers=2, use_cuda=False):
        super(MetaLearner, self).__init__()
        self.name = "default"
        self.opt_wrapper = func
        self.hidden_size = num_hidden
        self.use_cuda = use_cuda
        self.num_params = self.opt_wrapper.optimizee.parameter.data.view(-1).size(0)

        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.ln1 = LayerNorm1D(num_hidden)
        self.lstms = nn.ModuleList()
        # self.lstms = []
        for i in range(num_layers):
            self.lstms.append(LayerNormLSTMCell(num_hidden, num_hidden))
            # self.lstms.append(nn.LSTMCell(num_hidden, num_hidden))

        self.linear_out = nn.Linear(num_hidden, 1)

        # Linear transformation to generate pi(t=k|grad x, T)
        # self.linear_pi = nn.Linear(num_hidden, 1, bias=bias)
        # self.linear_future = nn.Linear(num_hidden, 1, bias=bias)
        # variable used to
        # self.loss_meta_model = None
        # array that collects loss components that we need for the final update of the T-optimizer
        self.q_t = None

    def cuda(self):
        super(MetaLearner, self).cuda()
        for i in range(len(self.lstms)):
            self.lstms[i].cuda()

    def zero_grad(self):
        super(MetaLearner, self).zero_grad()
        # make sure we also reset the gradients of the LSTM cells as well
        for i, cells in enumerate(self.lstms):
            cells.zero_grad()

    def reset_lstm(self, func, keep_states=False):
        # copy the quadratic function "func" to our internal quadratic function we keep in the meta optimizer
        self.opt_wrapper.reset(func)
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
        if self.use_cuda:
            x = x.cuda()

        x_out = []
        for t in np.arange(self.num_params):
            x_t = x[t].unsqueeze(1)
            x_t = F.tanh(self.ln1(self.linear1(x_t)))
            for i in range(len(self.lstms)):
                if x_t.size(0) != self.hx[t][i].size(0):
                    self.hx[t][i] = self.hx[t][i].expand(x_t.size(0), self.hx[t][i].size(1))
                    self.cx[t][i] = self.cx[t][i].expand(x_t.size(0), self.cx[t][i].size(1))

                self.hx[t][i], self.cx[t][i] = self.lstms[i](x_t, (self.hx[t][i], self.cx[t][i]))
                x_t = self.hx[t][i]
            x_t = self.linear_out(x_t)
            x_out.append(x_t)

        res = torch.cat(x_out)

        return res

    def forward_old(self, x):
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

    def meta_update(self, func_with_grads, loss_type="MSE"):
        """
            We use the quadratic loss surface - which is a copy of the "outside" loss surface that we're trying to optimize
            - to (a) compute the new parameters of this loss surface by taking the gradients of the outside loss surface
            pass them through our LSTM and subtract (gradient descent) the LSTM output from the "inside" loss surface
            parameters.
            (b) we copy the new parameter of the "inside" loss surface
            :param func_with_grads:
            :return:
        """
        parameters = Variable(self.opt_wrapper.optimizee.parameter.data)
        grads = Variable(func_with_grads.parameter.grad.data)
        if self.use_cuda:
            parameters = parameters.cuda()

        parameters = parameters - self(grads)
        # copy updated parameters to our "inside" loss surface model
        self.opt_wrapper.set_parameters(parameters)
        # copy new parameters also to "outside" func that deliverd the grads in the first place
        self.opt_wrapper.copy_params_to(func_with_grads)
        if loss_type == "EVAL":
            loss = torch.sum(self.opt_wrapper.optimizee.func(parameters))
        elif loss_type == "MSE":
            loss = 0.5 * torch.sum((self.opt_wrapper.optimizee.true_opt - parameters) ** 2)
        else:
            raise ValueError("<{}> is not a valid option for loss_type.".format(loss_type))

        return loss
        # return parameters

    def loss_func(self):
        assert self.loss_meta_model.size()[0] == self.q_t.size()[0], "Length of two objects is not the same!"
        q_soft = F.softmax(self.q_t)
        # kl =
        return torch.mean(q_soft.mul(self.loss_meta_model))

    @property
    def sum_grads(self):
        sum_grads = 0
        for i, cells in enumerate(self.lstms):
            for j, module in enumerate(cells.children()):
                for param_name, params in module._parameters.iteritems():
                    sum_grads += torch.sum(params.grad.data)
        for param in self.parameters():
            sum_grads += torch.sum(param.grad.data)

        return sum_grads


class WrapperOptimizee(object):
    """
        this class holds the quadratic function that we want to minimize by means of the meta-optimizer
        we need to reset
    """
    def __init__(self, func):
        self.optimizee = func

    def reset(self, q_func):
        self.optimizee.parameter = Variable(q_func.parameter.data, requires_grad=True)
        self.optimizee.x_min = q_func.x_min
        self.optimizee.x_max = q_func.x_max
        self.optimizee.func = q_func.f
        self.optimizee.true_opt = q_func.true_opt

    def set_parameters(self, parameters):
        self.optimizee.parameter.data.copy_(parameters.data)

    def copy_params_to(self, func):
        # copy parameter from the meta_model function we just updated in meta_update to the
        # function that delivered the gradients...call it the "outside" function
        func.parameter.data.copy_(self.optimizee.parameter.data)
