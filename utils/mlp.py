import copy
from collections import OrderedDict
from functools import reduce
from operator import mul

import torch
import torch.nn as nn
from torch.autograd import Variable


class MLP(nn.Module):

    def __init__(self, mlp_architecture, make_copy_obj=True):
        super(MLP, self).__init__()
        self.nn_architecture = mlp_architecture
        self._generate_network()
        self.initial_params = self.get_flat_params().clone()
        if make_copy_obj:
            self.eval_copy = self._make_sequential_module()

        else:
            self.eval_copy = None
        # slightly awkward, we need this attribute in order to make the optimization process general. Remember for the
        # other experiments (regression) we work with batches of function
        self.num_of_funcs = 1

    def _generate_network(self):
        self.input_layer = nn.Linear(self.nn_architecture["n_input"], self.nn_architecture["n_hidden_layer1"])
        self.hidden_layer1 = nn.Linear(self.nn_architecture["n_hidden_layer1"], self.nn_architecture["n_output"])
        self.act_output_layer = getattr(torch.nn, self.nn_architecture["act_func_output"])()
        self.loss_function = getattr(torch.nn, self.nn_architecture["loss_function"])()

    def forward(self, inputs):
        x = inputs.view(-1, self.nn_architecture["n_input"])
        out = self.input_layer(x)
        out = self.hidden_layer1(out)
        out = self.act_output_layer(out)

        return out

    def compute_loss(self, y_pred, y_true):
        loss = self.loss_function(y_pred, y_true)
        return loss

    def evaluate(self, x, use_copy_obj=False, compute_loss=False, y_true=None):
        """

        :param x: the MNIST image passed as an autograd Variable, can have 3 dim or 2 dim (2 & 3 combined)
        :param use_copy_obj: set to True in case we want to evaluate the copy object
        :param compute_loss: returns the loss if set to true
        :param y_true: the true labels, REQUIRED if compute_loss is set to True
        :return: y_pred (logits, torch.FloatTensor) or loss (scaler)
        """
        if not use_copy_obj:
            y_pred = self.forward(x)
        else:
            if x.dim() > 2:
                x = x.view(-1, self.nn_architecture["n_input"])
            y_pred = self.eval_copy(x)

        if compute_loss:
            if y_true is None:
                raise ValueError("Can't compute loss if y_true parameter is None")
            else:
                return self.compute_loss(y_pred, y_true)
        else:
            return y_pred

    @staticmethod
    def accuracy(y_pred, y_true):
        _, predicted = torch.max(y_pred.data, 1)
        correct = (predicted == y_true).sum()
        return correct

    def get_flat_params(self):
        params = []

        for name, module in self.named_children():
            if name != "eval_copy":
                for p_name, param in module.named_parameters():
                    params.append(param.view(-1))

        return torch.cat(params).unsqueeze(1)

    def get_flat_grads(self):
        grads = []

        for name, module in self.named_children():
            if name != "eval_copy":
                for p_name, param in module.named_parameters():
                    if param.grad is not None:
                        grads.append(param.grad.view(-1))
                    else:
                        print("param-name {}.{} no gradients".format(name, p_name))

        return torch.cat(grads).unsqueeze(1)

    def set_parameters(self, flat_params):
        offset = 0
        for name, module in self.named_children():
            if name != "eval_copy":
                for p_name, param in module.named_parameters():
                    param_shape = param.size()
                    param_flat_size = reduce(mul, param_shape, 1)
                    param.data.copy_(flat_params.data[offset:offset + param_flat_size].view(*param_shape))
                    offset += param_flat_size

    def set_eval_obj_parameters(self, flat_params):

        offset = 0
        for name, module in self.eval_copy.named_children():
                for p_name, param in module.named_parameters():
                    param_shape = param.size()
                    param_flat_size = reduce(mul, param_shape, 1)
                    module._parameters[p_name] = flat_params[offset:offset + param_flat_size].view(*param_shape)
                    offset += param_flat_size

    def _make_sequential_module(self):
        """
        we make a deep copy of the modules of the network (EXCEPT FOR THE LOSS FUNCTION) and use this object
        to evaluate the network. Unfortunately this is necessary otherwise the meta update won't work
        :return: sequential torch.nn object that contains the network layers as a sequence...in fact a copy of the net
        """
        return nn.Sequential(OrderedDict([("copy_"+m_name, copy.deepcopy(module)) for m_name, module in self.named_children()
                                          if m_name != "loss_function"]))

    def reset(self):
        reset_params = self.initial_params.data.clone()
        self.set_parameters(Variable(reset_params))
        if self.eval_copy is not None:
            self.set_eval_obj_parameters(Variable(reset_params))

    def reset_params(self):
        # break the backward chain by declaring param Variable again with same Tensor values
        # print(">>>>>>>>> RESETTING PARAMETER IN ORDER TO BREAK BACKWARD() OPERATION IN MLP object <<<<<<<<<")
        for module_name, module in self.named_children():
            if module_name != "eval_copy":
                for p_name, param in module.named_parameters():
                    module._parameters[p_name] = Variable(module._parameters[p_name].data, requires_grad=True)

    def test_model(self, dataset, use_cuda=False):
        # Test the Model
        correct = 0
        total = 0
        i = 0

        for images, labels in dataset.test_loader:
            if use_cuda:
                images = images.cuda()
                labels = labels.cuda()
            i += 1
            images = Variable(images.view(-1, self.nn_architecture["n_input"]))
            outputs = self.evaluate(images, use_copy_obj=True)
            total += labels.size(0)
            correct += self.accuracy(outputs, labels)

        return 100. * correct / float(total)


class Net(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(Net, self).__init__()
        self.input_size = input_size
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_size)
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out

    def get_flat_grads(self):
        grads = []

        for name, module in self.named_children():
            if name != "eval_copy":
                for p_name, param in module.named_parameters():
                    print("param {}".format(p_name))
                    if param.grad is not None:
                        grads.append(param.grad.view(-1))
                    else:
                        print("no gradients")