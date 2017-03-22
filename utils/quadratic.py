import numpy as np
from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig
from datetime import datetime

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
        self.W = Variable(torch.randn(5) * 8, requires_grad=False)
        self.W.data.abs_()

        while np.any(self.W.data.numpy()) == 0 or not (x_min < self.W[0].data.numpy()[0] < x_max) or \
                not (x_min < self.W[1].data.numpy()[0] < x_max):
            self.W = Variable(torch.randn(5) * 4, requires_grad=False)
            self.W.data.abs_()

        self.x_min = x_min
        self.x_max = x_max
        self.func = self.f
        self.value_hist = {'x1': [], 'x2': [], 'y': []}
        self.parameter = Variable(self._sample_x0(), requires_grad=True)
        self.true_opt = self._get_true_min_f()

        if self.use_cuda:
            self.W = self.W.cuda()
            self.true_opt = self.true_opt.cuda()
            self.parameter = self.parameter.cuda()

    def _get_true_min_f(self):
        return Variable(torch.cat((self.W[0].data, self.W[1].data), 0).unsqueeze(1),
                        requires_grad=False)

    def _sample_x0(self):
        x1_val = torch.FloatTensor(1)
        x2_val = torch.FloatTensor(1)
        x1_val = self.x_min + (self.x_max - self.x_min) * torch.FloatTensor.uniform_(x1_val)
        x2_val = self.x_min + (self.x_max - self.x_min) * torch.FloatTensor.uniform_(x2_val)

        x0 = torch.FloatTensor(torch.cat((x1_val, x2_val), 0))
        if self.use_cuda:
            x0 = x0.cuda()
        x0 = x0.unsqueeze(1)

        return x0

    def f(self, X):

        y = ((X[0] - self.W[0].expand_as(X[0])) ** 2 / self.W[2].expand_as(X[0]) ** 2 +
            (X[1] - self.W[1].expand_as(X[1])) ** 2 / self.W[2].expand_as(X[1]) ** 2 + self.W[3].expand_as(X[1])) * \
            self.W[4].expand_as(X[0])

        return y.unsqueeze(1)

    def f_at_xt(self, hist=False):
        y = self.func(self.parameter)
        if hist:
            self.value_hist['x1'].append(self.parameter[0].squeeze().cpu().data.clone().numpy()[0])
            self.value_hist['x2'].append(self.parameter[0].squeeze().cpu().data.clone().numpy()[0])
            self.value_hist['y'].append(y.squeeze().cpu().data.numpy()[0])
        return y.squeeze()

    @property
    def min_value(self):
        return self.f(self.true_opt).squeeze()

    @property
    def poly_desc(self):
        """Creates a string description of a quadratic."""
        result = "(x1-{:.2f})^2/{:.2f}^2".format(self.W[0].cpu().data.numpy()[0],
                                                            self.W[2].cpu().data.numpy()[0])
        result += "+(x2-{:.2f})^2/{:.2f}^2".format(self.W[1].cpu().data.numpy()[0],
                                                              self.W[2].cpu().data.numpy()[0])
        result += "+{:.2f})*{:.2f}".format(self.W[3].cpu().data.numpy()[0], self.W[4].cpu().data.numpy()[0])

        return result

    def plot_func(self, fig_name=None, height=8, width=6, do_save=False, show=False):
        steps = (self.x_max - self.x_min) * 25
        x_range = np.linspace(self.x_min, self.x_max, steps)
        X, Y = np.meshgrid(x_range, x_range)
        f_in = Variable(torch.FloatTensor(torch.cat((torch.from_numpy(X.ravel()).float(), torch.from_numpy(Y.ravel()).float()), 1).t()))
        Z = self.f(f_in)
        plt.figure(figsize=(height, width))
        plt.contourf(X, Y, Z.data.view(steps, steps).numpy(), steps)
        plt.plot(self.true_opt[0].data.numpy(), self.true_opt[1].data.numpy(), 'yo')
        # print(self.value_hist['x1'], self.value_hist['x2'])
        plt.plot(self.value_hist['x1'], self.value_hist['x2'], 'rx-', markersize=5)
        plt.title(self.poly_desc)

        if do_save:
            if fig_name is None:
                fig_name = datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
                fig_name = "figures/" + fig_name + ".png"
            plt.savefig(fig_name, bbox_inches='tight')
            print("INFO - Successfully saved fig %s" % fig_name)

        if show:
            plt.show()

        plt.close()
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
