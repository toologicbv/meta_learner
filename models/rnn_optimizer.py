
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from layer_norm_lstm import LayerNormLSTMCell
from layer_norm import LayerNorm1D


class MetaLearner(nn.Module):

    def __init__(self, func, num_inputs=1, num_hidden=20, num_layers=2, use_cuda=False):
        super(MetaLearner, self).__init__()
        self.name = "default"
        self.opt_wrapper = func
        self.hidden_size = num_hidden
        self.use_cuda = use_cuda
        self.num_params = self.opt_wrapper.optimizee.parameter.data.view(-1).size(0)

        self.linear1 = nn.Linear(num_inputs, num_hidden)
        self.ln1 = LayerNorm1D(num_hidden)
        self.lstms = nn.ModuleList()
        for i in range(num_layers):
            self.lstms.append(LayerNormLSTMCell(num_hidden, num_hidden))
            # self.lstms.append(nn.LSTMCell(num_hidden, num_hidden))

        self.linear_out = nn.Linear(num_hidden, 1)

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


class AdaptiveMetaLearner(MetaLearner):

    def __init__(self, func, num_inputs=1, num_hidden=20, num_layers=2, use_cuda=False):
        super(AdaptiveMetaLearner, self).__init__(func, num_inputs, num_hidden, num_layers, use_cuda)
        self.linear_grads = nn.Linear(self.num_params, 1)
        # holds losses from each optimizer step
        self.losses = []
        self.q_t = []
        # number of parameters extend by one for the WEIGHT factor per timestep
        self.num_params = self.opt_wrapper.optimizee.parameter.data.view(-1).size(0) + 1

    def forward(self, x):
        # extend the input by one, to start with we take the mean of the incoming gradients
        mean_grad = torch.mean(x)
        x = torch.cat((x, mean_grad))
        # qt = Variable(torch.zeros(1))
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
            if t != self.num_params - 1:
                x_out.append(x_t)
            else:
                qt = x_t

        x_out = torch.cat(x_out)

        return tuple((x_out, qt))

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

        delta_grads, qt = self(grads)
        parameters = parameters - delta_grads
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
        # collect the parts we need to calculate the final loss over all timesteps

        self.losses.append(Variable(loss.data))
        self.q_t.append(qt)

        return loss

    def final_loss(self):
        assert len(self.losses) == len(self.q_t), "Length of two objects is not the same!"
        losses = torch.cat(self.losses, 0)
        q_t = torch.cat(self.q_t, 0)
        q_soft = F.softmax(q_t)

        return torch.mean(q_soft * losses)

    def reset_final_loss(self):
        self.losses = []
        self.q_t = []