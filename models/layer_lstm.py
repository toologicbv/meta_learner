import torch.nn as nn
import torch.nn.functional as F


class LayerLSTMCell(nn.Module):

    def __init__(self, num_inputs, num_hidden, forget_gate_bias=0):
        super(LayerLSTMCell, self).__init__()

        self.forget_gate_bias = forget_gate_bias
        self.num_hidden = num_hidden
        self.fc_i2h = nn.Linear(num_inputs, 4 * num_hidden)
        self.fc_h2h = nn.Linear(num_hidden, 4 * num_hidden)

    def forward(self, inputs, state):
        hx, cx = state
        # i2h = self.fc_i2h(inputs)
        # h2h = self.fc_h2h(hx)
        x = self.fc_i2h(inputs) + self.fc_h2h(hx)
        gates = x.split(self.num_hidden, 1)

        in_gate = F.sigmoid(gates[0])
        forget_gate = F.sigmoid(gates[1] + self.forget_gate_bias)
        out_gate = F.sigmoid(gates[2])
        cell_gate = F.tanh(gates[3])

        cx = forget_gate * cx + in_gate * cell_gate
        # hx = out_gate * F.tanh(self.ln_h2o(cx))
        hx = out_gate * F.tanh(cx)
        return hx, cx
