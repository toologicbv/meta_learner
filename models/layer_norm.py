import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class LayerNorm1D(nn.Module):

    def __init__(self, num_outputs, eps=1e-5, bias=True):
        super(LayerNorm1D, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_outputs))
        if bias:
            self.bias = nn.Parameter(torch.zeros(1, num_outputs))
        else:
            self.register_parameter('bias', None)

    def forward(self, inputs):
        input_mean = inputs.mean(1).expand_as(inputs)
        input_std = inputs.std(1).expand_as(inputs)
        x = (inputs - input_mean) / (input_std + self.eps)
        if self.bias is not None:
            return x * self.weight.expand_as(x) + self.bias.expand_as(x)
        else:
            return x * self.weight.expand_as(x)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)
