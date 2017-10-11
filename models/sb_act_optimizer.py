
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from layer_lstm import LayerLSTMCell


class StickBreakingACTBaseModel(nn.Module):

    def __init__(self, num_inputs=1, num_hidden=20, num_layers=2, use_cuda=False,
                 output_bias=False, fg_bias=1., custom=True):
        super(StickBreakingACTBaseModel, self).__init__()
        self.name = "default"
        self.hidden_size = num_hidden
        self.use_cuda = use_cuda
        self.hx = None
        self.cx = None
        self.num_layers = num_layers
        self.linear_in = nn.Linear(num_inputs, num_hidden)
        self.lstms = nn.ModuleList()
        for i in range(self.num_layers):
            if custom:
                self.lstms.append(LayerLSTMCell(num_hidden, num_hidden, forget_gate_bias=fg_bias))
            else:
                self.lstms.append(nn.LSTMCell(num_hidden, num_hidden))

        self.theta_linear_out = nn.Linear(num_hidden, 1, bias=output_bias)
        self.rho_linear_out = nn.Linear(num_hidden, 1, bias=True)
        # lower the initial bias value of the linear output layer that generates the rho_t values, this helps the model
        # to explore longer time sequences
        self.rho_linear_out.bias.data = torch.FloatTensor([-1.])
        self.losses = []
        self.qt = []
        self.q_soft = None
        self.epochs_trained = 0
        if self.use_cuda:
            self.cuda()

    def forward(self, x_t):
        if self.use_cuda and not x_t.is_cuda:
            x_t = x_t.cuda()
        if x_t.dim() == 1:
            x_t = x_t.unsqueeze(1)
        # the 2nd input dimension is 3 when we're optimizing the MLPs. In this case we scale the outputs as
        # mentioned in the L2L paper by 0.1
        if x_t.size(1) == 3:
            do_scale = True
        else:
            do_scale = False

        x_t = self.linear_in(x_t)
        for i in range(len(self.lstms)):
            if x_t.size(0) != self.hx[i].size(0):
                self.hx[i] = self.hx[i].expand(x_t.size(0), self.hx[i].size(1))
                self.cx[i] = self.cx[i].expand(x_t.size(0), self.cx[i].size(1))

            self.hx[i], self.cx[i] = self.lstms[i](x_t, (self.hx[i], self.cx[i]))
            x_t = self.hx[i]

        theta_out = self.theta_linear_out(x_t)
        rho_out = F.sigmoid(self.rho_linear_out(x_t))
        if do_scale:
            # in case of MLP we scale the output as mentioned in L2L paper
            theta_out = 0.1 * theta_out

        return tuple((theta_out.squeeze(), rho_out.squeeze()))

    def final_loss(self):

        losses = torch.cat(self.losses, 1)
        return torch.mean(losses)

    def reset_final_loss(self):
        self.losses = []
        self.qt = []
        self.q_soft = None

    def cuda(self):
        super(StickBreakingACTBaseModel, self).cuda()

    def zero_grad(self):
        super(StickBreakingACTBaseModel, self).zero_grad()

    def save_params(self, absolute_path):
        torch.save(self.state_dict(), absolute_path)

    def reset_lstm(self, keep_states=False):

        if keep_states:
            for i in range(self.num_layers):
                self.hx[i] = Variable(self.hx[i].data)
                self.cx[i] = Variable(self.cx[i].data)
        else:
            self.hx = []
            self.cx = []
            for i in range(self.num_layers):
                self.hx.append(Variable(torch.zeros(1, self.hidden_size)))
                self.cx.append(Variable(torch.zeros(1, self.hidden_size)))
                if self.use_cuda:
                    self.hx[i], self.cx[i] = self.hx[i].cuda(), self.cx[i].cuda()

    def sum_grads(self, verbose=False):
        sum_grads = 0
        for name, param in self.named_parameters():
            if param.grad is not None:
                sum_grads += torch.sum(torch.abs(param.grad.data))
            else:
                if verbose:
                    print("WARNING - No gradients for parameter >>> {} <<<".format(name))

        return sum_grads
