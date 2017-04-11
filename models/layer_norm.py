import torch
import torch.nn as nn
import math


class LayerNorm1D(nn.Module):

    def __init__(self, num_outputs, eps=1e-5, use_bias=True):
        super(LayerNorm1D, self).__init__()
        self.use_bias = use_bias
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, num_outputs))
        # added this Tuesday 11-4 when experimenting with biases of LSTM
        stdv = 1.0 / math.sqrt(num_outputs)
        self.weight.data.uniform_(-stdv, stdv)

        if use_bias:
            self.bias = nn.Parameter(torch.zeros(1, num_outputs))

    def forward(self, inputs):
        input_mean = inputs.mean(1).expand_as(inputs)
        input_std = inputs.std(1).expand_as(inputs)
        x = (inputs - input_mean) / (input_std + self.eps)
        if self.use_bias:
            out = x * self.weight.expand_as(x) + self.bias.expand_as(x)
        else:
            out = x * self.weight.expand_as(x)

        return out
