from torch.autograd import Variable
import torch
import matplotlib.pyplot as plt
from matplotlib.pylab import savefig

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
        return y

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

    def plot_func(self, fig_name=None, height=8, width=6, show=False):
        plt.figure(figsize=(height, width))
        # print(self.value_hist['x'][0:10])
        # print(self.value_hist['y'][0:10])
        plt.plot(self.x_range.data.view(-1).numpy(), self.y_range.data.view(-1).numpy(), 'r-')
        if len(self.value_hist['x']) != 0:
            plt.plot(self.value_hist['x'], self.value_hist['y'], 'b+')
        plt.title("Quadratic function %s" % self.poly_desc())
        if fig_name is None:
            fig_name = self.poly_desc() + ".png"
        savefig(fig_name, bbox_inches='tight')
        if show:
            plt.show()
