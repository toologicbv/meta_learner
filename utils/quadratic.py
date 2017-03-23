import os

import numpy as np
from torch.autograd import Variable
import torch
from torch.nn import init
import matplotlib.pyplot as plt
from cycler import cycler

from datetime import datetime
from config import config

POLY_DEGREE = 2


class Quadratic(object):

    def __init__(self, x_min=-6, x_max=6, use_cuda=False):
        self.use_cuda = use_cuda
        self.W = Variable(torch.zeros(POLY_DEGREE, 1))

        while self.W[0].data.numpy()[0] == 0:
            self.W = Variable(torch.randn(POLY_DEGREE, 1) * 0.5, requires_grad=False)
            self.W[0].data.abs_()
            self.b = Variable(torch.randn(1) * 20, requires_grad=False)
            self.b.data.abs_()

        self.x_min = x_min
        self.x_max = x_max
        self.func = self.f
        self.value_hist = {'x': [], 'y': []}
        self.true_opt = self._get_true_min_f()
        self.parameter = Variable(self._sample_x0(), requires_grad=True)
        if self.use_cuda:
            self.W = self.W.cuda()
            self.b = self.b.cuda()
            self.true_opt = self.true_opt.cuda()
            self.parameter = self.parameter.cuda()

    def f_at_xt(self, hist=False):
        y = self.func(self.parameter)
        if hist:
            self.value_hist['x'].append(self.parameter.squeeze().cpu().data.clone().numpy()[0])
            self.value_hist['y'].append(y.squeeze().cpu().data.numpy()[0])
        return y.squeeze()

    def f(self, x):
        X = torch.cat((x.pow(2), x), 1)
        Y = X.mm(self.W) + self.b.unsqueeze(1).expand_as(x)

        return Y

    def _get_true_min_f(self):

        self.x_range = Variable(torch.linspace(self.x_min, self.x_max, 1000).unsqueeze(1))
        if self.use_cuda:
            self.x_range.data.cuda()

        self.y_range = self.f(self.x_range)
        y_min, x1_idx = torch.min(self.y_range, 0)
        x_true_min = torch.gather(self.x_range, 0, x1_idx)

        return x_true_min

    def _sample_x0(self):
        x_val = torch.FloatTensor(1)
        x_val = self.x_min + (self.x_max - self.x_min) * torch.FloatTensor.uniform_(x_val)
        if self.use_cuda:
            x_val = x_val.cuda()

        return x_val.unsqueeze(1)

    def poly_desc(self):
        """Creates a string description of a polynomial."""
        result = 'y = '
        for i, w in enumerate(self.W):
            result += '{:+.2f} x^{} '.format(w.cpu().data.numpy()[0], len(self.W) - i)
        result += '{:+.2f}'.format(self.b.cpu().data.numpy()[0])
        return result

    def plot_func(self, fig_name=None, height=8, width=6, do_save=False, show=False):
        plt.figure(figsize=(height, width))
        # print(self.value_hist['x'][0:10])
        # print(self.value_hist['y'][0:10])
        plt.plot(self.x_range.data.view(-1).numpy(), self.y_range.data.view(-1).numpy(), 'r-')
        if len(self.value_hist['x']) != 0:
            plt.plot(self.value_hist['x'], self.value_hist['y'], 'b+')
        plt.title("Quadratic function %s" % self.poly_desc())
        if do_save:
            if fig_name is None:
                fig_name = self.poly_desc() + ".png"
            plt.savefig(fig_name, bbox_inches='tight')
        if show:
            plt.show()


class Quadratic2D(object):

    def __init__(self, x_min=-10, x_max=10, use_cuda=False):
        self.use_cuda = use_cuda
        # self.W = Variable(torch.randn(5) * 8, requires_grad=False)
        self.W = Variable(torch.zeros(5))
        init.normal(self.W)
        self.W.data.abs_()

        while torch.sum(torch.eq(self.W, 0)).data.numpy()[0] != 0 or \
                not (x_min < self.W[0].data.numpy()[0] < x_max) or \
                not (x_min < self.W[1].data.numpy()[0] < x_max) or \
                torch.sum(torch.le(self.W, 0.1)).data.numpy()[0] != 0:
            init.normal(self.W)
            self.W.data.abs_()

        self.x_min = x_min
        self.x_max = x_max
        self.func = self.f
        self.value_hist = {'x1': [], 'x2': [], 'y': []}
        self.parameter = Variable(self._sample_x0(), requires_grad=True)
        self.initial_parms = self.parameter.clone()
        self.true_opt = self._get_true_min_f()
        self.error = torch.zeros(1)

    def reset(self):
        self.value_hist = {'x1': [], 'x2': [], 'y': []}
        reset_params = self.initial_parms.data.clone()
        self.parameter = Variable(reset_params, requires_grad=True)

    def _get_true_min_f(self):
        return Variable(torch.cat((self.W[0].data, self.W[1].data), 0).unsqueeze(1),
                        requires_grad=False)

    def _sample_x0(self):
        x1_val = torch.FloatTensor(1)
        x2_val = torch.FloatTensor(1)
        x1_val = self.x_min + (self.x_max - self.x_min) * torch.FloatTensor.uniform_(x1_val)
        x2_val = self.x_min + (self.x_max - self.x_min) * torch.FloatTensor.uniform_(x2_val)
        while not (self.x_min <= x1_val[0] <= self.x_max and self.x_min <= x2_val[0] <= self.x_max
                   and (torch.abs(x1_val - self.W[0].data)[0] > 4.)
                   and (torch.abs(x2_val - self.W[1].data)[0] > 4.)):
            x1_val = self.x_min + (self.x_max - self.x_min) * torch.FloatTensor.uniform_(x1_val)
            x2_val = self.x_min + (self.x_max - self.x_min) * torch.FloatTensor.uniform_(x2_val)

        x0 = torch.FloatTensor(torch.cat((x1_val, x2_val), 0))
        x0 = x0.unsqueeze(1)

        return x0

    def f(self, X):
        if self.use_cuda:
            X = X.cuda()
            self.W = self.W.cuda()

        y = ((X[0] - self.W[0].expand_as(X[0])) ** 2 / self.W[2].expand_as(X[0]) ** 2 +
            (X[1] - self.W[1].expand_as(X[1])) ** 2 / self.W[2].expand_as(X[1]) ** 2 + self.W[3].expand_as(X[1]))
        # omitting the last parameter after figuring out that this made the functions pretty hard to optimize
        # * self.W[4].expand_as(X[0])

        return y.unsqueeze(1)

    def f_at_xt(self, hist=False):
        y = self.func(self.parameter)
        if hist:
            self.value_hist['x1'].append(self.parameter[0].squeeze().cpu().data.clone().numpy()[0])
            self.value_hist['x2'].append(self.parameter[1].squeeze().cpu().data.clone().numpy()[0])
            self.value_hist['y'].append(y.squeeze().cpu().data.numpy()[0])
        return y.squeeze()

    @property
    def min_value(self):
        return self.f(self.true_opt).squeeze()

    @property
    def poly_desc(self):
        """Creates a string description of a quadratic."""
        result = "((x1-{:.2f})^2/{:.2f}^2".format(self.W[0].cpu().data.numpy()[0],
                                                            self.W[2].cpu().data.numpy()[0])
        result += "+(x2-{:.2f})^2/{:.2f}^2)".format(self.W[1].cpu().data.numpy()[0],
                                                              self.W[2].cpu().data.numpy()[0])
        result += "+{:.2f}".format(self.W[3].cpu().data.numpy()[0])
        # result += "+{:.2f})*{:.2f}".format(self.W[3].cpu().data.numpy()[0], self.W[4].cpu().data.numpy()[0])

        return result

    def plot_func(self, fig_name=None, height=8, width=6, do_save=False, show=False):

        cm = plt.get_cmap(config.color_map)
        steps = (self.x_max - self.x_min) * 25
        x_range = np.linspace(self.x_min, self.x_max, steps)
        X, Y = np.meshgrid(x_range, x_range)
        f_in = Variable(torch.FloatTensor(torch.cat((torch.from_numpy(X.ravel()).float(), torch.from_numpy(Y.ravel()).float()), 1).t()))
        Z = self.f(f_in)
        plt.figure(figsize=(height, width))
        plt.contourf(X, Y, Z.data.cpu().view(steps, steps).numpy(), steps)
        plt.plot(self.true_opt[0].data.cpu().numpy(), self.true_opt[1].data.cpu().numpy(), 'ro')
        # plot the gradient steps (lines) in different colors with increased transparency
        ax = plt.gca()
        num_points = len(self.value_hist['x1'])
        ax.set_prop_cycle(cycler('color', [cm(1. * i / (num_points - 1)) for i in range(num_points - 1)]))
        for i in range(num_points - 1):
            ax.plot(self.value_hist['x1'][i:i + 2], self.value_hist['x2'][i:i + 2], 'o-')
            # plt.plot(self.value_hist['x1'], self.value_hist['x2'], 'o-')
        plt.title(self.poly_desc)

        if do_save:
            dt = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
            if fig_name is None:
                fig_name = dt + ".png"
            else:
                fig_name = fig_name + "_" + dt + ".png"

            plt.savefig(fig_name, bbox_inches='tight')
            print("INFO - Successfully saved fig %s" % fig_name)

        if show:
            plt.show()

        plt.close()

    def compute_error(self):
        self.error = 0.5 * torch.sum((self.true_opt.data - self.parameter.data)**2)
        return self.error


class SimpleQuadratic(object):

    def __init__(self, use_cuda=False):
        self.x_min = -10
        self.x_max = 10
        self.use_cuda = use_cuda
        self.W = Variable(torch.zeros(2, 2), requires_grad=False)
        self.b = Variable(torch.zeros(2, 1), requires_grad=False)
        init.normal(self.W)
        init.normal(self.b)
        self.func = self.f
        self.value_hist = {'x1': [], 'x2': [], 'y': []}
        self.parameter = Variable(torch.zeros(2, 1), requires_grad=True)
        init.normal(self.parameter, mean=0, std=2.)
        self.initial_parms = Variable(torch.zeros(self.parameter.data.size()))
        self.initial_parms.data.copy_(self.parameter.data)
        self.true_opt = self._get_true_min_f()
        self.error = torch.zeros(1)

    def _get_true_min_f(self):
        x_true = np.dot(np.linalg.pinv(self.W.data.numpy()), self.b.data.numpy())
        return Variable(torch.from_numpy(x_true))

    def f(self, x):
        if self.use_cuda:
            x = x.cuda()
            self.W = self.W.cuda()
            self.b = self.b.cuda()

        y = torch.sum((x.t().mm(self.W) - self.b.expand_as(x))**2, 1)
        if y.dim() < 2:
            return y.unsqueeze(1)
        else:
            return y

    def f_at_xt(self, hist=False):
        y = self.func(self.parameter)
        if hist:
            self.value_hist['x1'].append(self.parameter[0].squeeze().cpu().data.clone().numpy()[0])
            self.value_hist['x2'].append(self.parameter[1].squeeze().cpu().data.clone().numpy()[0])
            self.value_hist['y'].append(y.squeeze().cpu().data.numpy()[0])
        return y.squeeze()

    @property
    def min_value(self):
        return self.f(self.true_opt).squeeze()

    @property
    def poly_desc(self):
        """Creates a string description of a quadratic."""
        result = "({:.2f}x1 - {:.2f})^2".format(self.W[0].cpu().data.numpy()[0],
                                                self.W[1].cpu().data.numpy()[0])

        return result

    def plot_func(self, fig_name=None, height=8, width=6, do_save=False, show=False):
        MAP = config.color_map
        cm = plt.get_cmap(MAP)
        steps = (self.x_max - self.x_min) * 25
        x_range = np.linspace(self.x_min, self.x_max, steps)
        X, Y = np.meshgrid(x_range, x_range)
        f_in = Variable(torch.FloatTensor(
            torch.cat((torch.from_numpy(X.ravel()).float(), torch.from_numpy(Y.ravel()).float()), 1).t()))
        Z = self.f(f_in)
        plt.figure(figsize=(height, width))
        plt.contourf(X, Y, Z.data.cpu().view(steps, steps).numpy(), steps)
        plt.plot(self.true_opt[0].data.cpu().numpy(), self.true_opt[1].data.cpu().numpy(), 'yo')
        # plot the gradient steps (lines) in different colors with increased transparency
        ax = plt.gca()
        num_points = len(self.value_hist['x1'])
        ax.set_prop_cycle(cycler('color', [cm(1. * i / (num_points - 1)) for i in range(num_points - 1)]))
        for i in range(num_points - 1):
            ax.plot(self.value_hist['x1'][i:i + 2], self.value_hist['x2'][i:i + 2], 'o-')
            # plt.plot(self.value_hist['x1'], self.value_hist['x2'], 'o-')
        plt.title(self.poly_desc)

        if do_save:
            if fig_name is None:
                fig_name = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                fig_name = os.path.join(config.figure_path, (fig_name + ".png"))
            plt.savefig(fig_name, bbox_inches='tight')
            print("INFO - Successfully saved fig %s" % fig_name)

        if show:
            plt.show()

        plt.close()

    def compute_error(self):
        self.error = 0.5 * torch.sum((self.true_opt.data - self.parameter.data) ** 2)
        return self.error



# q2d = Quadratic2D()
# print(q2d.poly_desc)
# print(q2d.parameter[0].data.numpy()[0], q2d.parameter[1].data.numpy()[0], q2d.f_at_xt(hist=True))
# print("True min ({:.2f}, {:.2f})".format(q2d.true_opt[0].data.numpy()[0], q2d.true_opt[1].data.numpy()[0]))
# print("q2d.f(q2d.parameter) ", type(q2d.f(q2d.parameter)))
# loss = q2d.f_at_xt(hist=True)
# print(type(loss), type(loss.data), type(loss.data[0]))

# q2d.parameter = q2d.parameter + Variable(torch.FloatTensor([1.3, 2.1]))
# _ = q2d.f_at_xt(hist=True)
# q2d.plot_func(show=False, do_save=False)
# q = Quadratic()
# loss = q.f_at_xt()
