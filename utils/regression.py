import numpy as np
from torch.autograd import Variable
import torch
from torch.nn import init
from collections import OrderedDict
from matplotlib import pyplot as plt

from datetime import datetime
from pytz import timezone
import os
from config import config
from quadratic import create_exp_label
from itertools import cycle
from cycler import cycler


def neg_log_likelihood_loss(true_y, est_y, stddev, N, sum_batch=False, size_average=False):
    """
    Negative log-likelihood calculation
    if sum_batch=False the function returns a tensor [batch_size, 1]
    otherwise a tensor of [1]

    :param true_y: true target values
    :param est_y:  estimated target values
    :param stddev: standard deviation of normal distribution
    :param N: number of samples
    :return: negative log-likelihood: ln p(y_true | x, weights, sigma^2)
    """
    if size_average:
        avg = 1/float(N)
    else:
        avg = 1.
    ll = avg * 0.5 * 1 / stddev**2 * \
         torch.sum((true_y - est_y) ** 2, 1) + \
           N / 2 * (np.log(stddev**2) + np.log(2 * np.pi))
    if sum_batch:
        # we also sum over the mini-batch, dimension 0
        ll = torch.sum(ll)

    return ll


def get_axis_ranges(init_params, true_params, delta=2, steps=100):
    if init_params > true_params:
        max_dim = init_params + delta
        min_dim = true_params - delta
    else:
        max_dim = true_params + delta
        min_dim = init_params - delta

    axis_range = np.linspace(min_dim, max_dim, steps)

    return axis_range


def get_true_params(loc=0., scale=4, size=(2, 1)):
    np_params = np.random.normal(loc=loc, scale=scale, size=size)
    return Variable(torch.FloatTensor(torch.from_numpy(np_params).float()), requires_grad=False)


def init_params(size=(1, 1)):
    theta = torch.FloatTensor(*size)
    init.uniform(theta, -0.1, 0.1)
    return theta


def get_noise(size=None, sigma=1.):
    if size is None:
        size = (1, 1)
    else:
        size = (1, size)
    noise = torch.FloatTensor(*size)
    init.normal(noise, std=sigma)
    return Variable(noise)


def add_noise(y, noise=None):
    if noise is None:
        noise = get_noise(y.size())
    return y + noise.expand_as(y)


def sample_points(n_samples=100, ndim=2, x_min=-2., x_max=2.):

    x_array = []
    x_array.append(np.ones((n_samples, 1)))
    for i in range(2, ndim + 1):
        x_numpy = np.random.uniform(x_min, x_max, n_samples)
        x_array.append(x_numpy[:, np.newaxis])

    X = np.concatenate(x_array, axis=1).T
    X = Variable(torch.from_numpy(X).float())
    return X


def get_f_values(X, params):
    Y = torch.mm(params, X)
    return Y


def insert_samples_dim(X, size_dim1):
    """

    :param X: tensor of size [batch_size, num_params] that need to be extended with size_dim1
    :param size_dim1: size of the dimension 1 to be inserted in X
    :return: X of size [batch_size, size_dim1, num_params]
    """

    X = X.unsqueeze(X.dim())
    X = X.expand(X.size(0), X.size(1), size_dim1)
    X = torch.transpose(X, 1, 2)
    return X


class L2LQuadratic(object):
    """Quadratic problem: f(x) = ||Wx - y||."""

    def __init__(self, batch_size=128, num_dims=10, stddev=0.01, use_cuda=False):
        """Builds loss graph."""
        self.num_of_funcs = batch_size
        self.noise_sigma = stddev
        self.use_cuda = use_cuda
        self.dim = num_dims
        self.n_samples = 10

        # Non-trainable variables.
        w = torch.FloatTensor(batch_size, num_dims, num_dims)
        self.W = Variable(init.uniform(w), requires_grad=False)
        # self.W = Variable(init.normal(w, mean=1., std=stddev), requires_grad=False)
        if self.use_cuda:
            self.W = self.W.cuda()

        y = torch.FloatTensor(batch_size, num_dims)
        self.y = Variable(init.uniform(y), requires_grad=False)
        # self.y = Variable(init.normal(y, mean=1., std=stddev), requires_grad=False)
        if self.use_cuda:
            self.y = self.y.cuda()

        # Trainable variable.
        # TODO...true parameters
        self.true_params = Variable(torch.zeros((batch_size, num_dims)))
        # self.true_params = Variable(self.get_true_params())
        if self.use_cuda:
            params = torch.FloatTensor(batch_size, num_dims).cuda()
            self.params = Variable(init.normal(params, mean=0., std=stddev).cuda(), requires_grad=True)
            self.true_params = self.true_params.cuda()
        else:
            params = torch.FloatTensor(batch_size, num_dims)
            self.params = Variable(init.normal(params, mean=0., std=stddev), requires_grad=True)

        self.initial_params = self.params.clone()

        self.param_hist = {}
        self._add_param_values(self.initial_params)

    def _add_param_values(self, parameters):
        self.param_hist[len(self.param_hist)] = parameters

    def set_parameters(self, parameters):
        self._add_param_values(parameters)
        self.params.data.copy_(parameters.data)

    def reset_params(self):
        # break the backward chain by declaring param Variable again with same Tensor values
        if self.use_cuda:
            self.params = Variable(self.params.data.cuda(), requires_grad=True)
        else:
            self.params = Variable(self.params.data, requires_grad=True)

    def reset(self):
        self.param_hist = {}
        reset_params = self.initial_params.data.clone()
        self.params = Variable(reset_params, requires_grad=True)
        self._add_param_values(self.initial_params)

    def compute_loss(self, average=True, params=None):
        if params is None:
            params = self.params
        product = torch.squeeze(torch.bmm(self.W, params.unsqueeze(params.dim())))
        if average:
            loss = torch.mean(torch.sum((product - self.y) ** 2, 1))
        else:
            loss = torch.sum((product - self.y) ** 2, 1)

        return loss

    def param_error(self, average=True):
        """
        computes parameter error and if applicable apply average over functions
        :param average:
        :return: parameter loss/error (Variable)
        """
        if average:
            N = float(self.num_of_funcs)
        else:
            N = 1
        param_error = 1/N * torch.sum((self.true_params - self.params) ** 2)
        return param_error.squeeze()

    def get_true_params(self):
        X = np.zeros((self.num_of_funcs, self.dim))

        for i in np.arange(self.num_of_funcs):
            A_plus = np.linalg.pinv(self.W.data.cpu().numpy()[i])
            b = self.y.data.cpu().numpy()[i]
            X[i, :] = np.squeeze(np.dot(A_plus, b))

        return torch.from_numpy(X).float()

    @property
    def min_loss(self):
        loss = self.compute_loss(params=self.true_params, average=True)
        print(loss.data.cpu().numpy())
        return loss


class RegressionFunction(object):

    def __init__(self, n_funcs=128, n_samples=10, param_sigma=4., stddev=0.01, x_dim=2, use_cuda=False):
        self.use_cuda = use_cuda
        self.stddev = stddev
        self.num_of_funcs = n_funcs
        self.n_samples = n_samples
        self.xdim = x_dim
        true_params = torch.FloatTensor(n_funcs, x_dim)
        self.true_params = Variable(init.uniform(true_params), requires_grad=False)

        if self.use_cuda:
            params = torch.FloatTensor(n_funcs, x_dim).cuda()
            self.params = Variable(init.normal(params, mean=0., std=stddev).cuda(), requires_grad=True)
            self.true_params = self.true_params.cuda()
        else:
            params = torch.FloatTensor(n_funcs, x_dim)
            self.params = Variable(init.normal(params, mean=0., std=stddev), requires_grad=True)

        self.initial_params = self.params.clone()

        w = torch.FloatTensor(self.num_of_funcs, self.n_samples, self.xdim)
        self.W = Variable(init.uniform(w), requires_grad=False)
        if self.use_cuda:
            self.W = self.W.cuda()

        self.y_no_noise = torch.squeeze(torch.bmm(self.W, self.true_params.unsqueeze(self.true_params.dim())))
        if self.num_of_funcs ==1:
            self.y_no_noise = self.y_no_noise.unsqueeze(0)
        noise = torch.FloatTensor(self.num_of_funcs, self.n_samples)
        self.noise = Variable(init.normal(noise, std=stddev), requires_grad=False)
        if self.use_cuda:
            self.noise = self.noise.cuda()

        self.y = self.y_no_noise + self.noise.expand_as(self.y_no_noise)
        if self.use_cuda:
            self.y = self.y.cuda()

        self.param_hist = {}
        self._add_param_values(self.initial_params)

    def set_parameters(self, parameters):
        self._add_param_values(parameters)
        self.params.data.copy_(parameters.data)

    def _add_param_values(self, parameters):
        self.param_hist[len(self.param_hist)] = parameters

    def copy_params_to(self, reg_func_obj):
        # copy parameter from the meta_model function we just updated in meta_update to the
        # function that delivered the gradients...call it the "outside" function
        reg_func_obj.params.data.copy_(self.params.data)

    def reset(self):
        self.param_hist = {}
        reset_params = self.initial_params.data.clone()
        if self.use_cuda:
            self.params = Variable(reset_params.cuda(), requires_grad=True)
        else:
            self.params = Variable(reset_params, requires_grad=True)

        self._add_param_values(self.initial_params)

    def reset_params(self):
        # break the backward chain by declaring param Variable again with same Tensor values
        if self.use_cuda:
            self.params = Variable(self.params.data.cuda(), requires_grad=True)
        else:
            self.params = Variable(self.params.data, requires_grad=True)

    def compute_loss(self, average=True, params=None):
        """
        compute mean squared loss and if applicable apply average over functions
        :param average:
        :return: loss (Variable)
        """
        if params is None:
            params = self.params
        if self.use_cuda and not params.cuda():
            params = params.cuda()
        # size of W = [batch, n_samples, xdim] and params = [batch, xdim]
        product = torch.squeeze(torch.bmm(self.W, params.unsqueeze(params.dim())))
        if self.num_of_funcs ==1:
            product = product.unsqueeze(0)

        if average:
            loss = torch.mean(torch.sum((product - self.y) ** 2, 1))
        else:
            loss = torch.sum((product - self.y) ** 2, 1)
        return loss

    def compute_neg_ll(self, average_over_funcs=False, size_average=False):
        ll = neg_log_likelihood_loss(self.y, self.y_t(), self.stddev, N=self.n_samples, sum_batch=False,
                                     size_average=size_average)
        if average_over_funcs:
            ll = torch.mean(ll, 0).squeeze()
        return ll

    def param_error(self, average=True):
        """
        computes parameter error and if applicable apply average over functions
        :param average:
        :return: parameter loss/error (Variable)
        """
        if average:
            N = float(self.num_of_funcs)
        else:
            N = 1
        param_error = 1/N * torch.sum((self.true_params - self.params) ** 2)
        return param_error.squeeze()

    def y_t(self, params=None):
        if params is None:
            params = self.params

        return torch.squeeze(torch.bmm(self.W, params.unsqueeze(params.dim())))

    def poly_desc(self, idx, f_true=True):
        if f_true:
            f_params = self.true_params.data.numpy()[idx, :]
        else:
            f_params = self.params.data.numpy()[idx, :]
        descr = r'${:.3} $'.format(f_params[0])
        for i in range(1, self.xdim):
            descr += r'$+ {:.3} x$ '.format(f_params[i])
        return descr

    def get_y_values(self, params):
        return torch.bmm(self.W, params)

    def plot_contour(self, f_idx=0, delta=2):
        np_init_params = self.initial_params[f_idx, :].data.numpy()
        torch_params = self.true_params[f_idx, :].unsqueeze(0)
        np_params = self.true_params[f_idx, :].data.numpy()
        y_true = get_f_values(torch_params, self.W[f_idx])
        steps = (delta + delta) * 25
        x_range = get_axis_ranges(np_init_params[0], np_params[0], delta=delta, steps=steps)
        y_range = get_axis_ranges(np_init_params[1], np_params[1], delta=delta, steps=steps)
        X, Y = np.meshgrid(x_range, y_range)
        f_in = Variable(torch.FloatTensor(torch.cat((torch.from_numpy(X.ravel()).float(),
                                                     torch.from_numpy(Y.ravel()).float()), 1)))
        Z_mean = get_f_values(self.W[f_idx], f_in)
        Y_true = y_true.expand_as(Z_mean)
        Z = add_noise(Z_mean, noise=self.noise)

        error = 0.5 * 1 / self.stddev**2 * torch.sum((Z - Y_true) ** 2, 1)

        plt.contourf(X, Y, error.data.cpu().view(steps, steps).numpy(), steps)

        return [x_range.min(), x_range.max(), y_range.min(), y_range.max()]

    def plot_func(self, f_idx, fig_name=None, height=8, width=6, do_save=False, show=False, exper=None, steps=None):
        p_colors = ['navajowhite', 'aqua', 'orange', 'red', 'yellowgreen', 'lightcoral', 'violet', 'dodgerblue',
                    'green', 'darkviolet']
        style = [[8, 4, 2, 4, 2, 4], [4, 2, 2, 4, 2, 4], [2, 2, 2, 4, 2, 4], [4, 8, 2, 4, 2, 4], [8, 8, 8, 4, 2, 4],
                 [4, 4, 2, 8, 2, 2]]
        iter_colors = cycle(p_colors)
        iter_styles = cycle(style)
        num_points = len(self.param_hist) - 1
        x = self.x.data.numpy()
        y_mean = self.y_no_noise.data.numpy()[f_idx, :]
        y_samples = self.y.data.numpy()[f_idx, :]
        y = self.y_t().data.numpy()[f_idx, :]
        f_true_desc = r'$f(x)=$' + self.poly_desc(f_idx) + r'$ (steps={})$'.format(num_points) + "\n"
        f_approx_desc = r'$\hat{f}(x)=$' + self.poly_desc(f_idx, f_true=False)
        l_title = f_true_desc + f_approx_desc
        plt.figure(figsize=(height, width))
        plt.title(l_title)

        plt.plot(x, y_mean, 'b-', lw=2, label=r'$f(x)$')
        plt.plot(x, y_samples, 'go', markersize=2, label=r'samples $\sigma={:.2}$'.format(self.stddev))
        if steps is not None:
            for i in steps:
                p_t = self.param_hist[i][f_idx, :].unsqueeze(0)
                y_t = torch.mm(self.W[f_idx], p_t).data.numpy().squeeze()
                c = iter_colors.next()
                # print("Color {}/{} ({:.3}/{:.3})".format(c, i, p_t.data.numpy()[0, 0], p_t.data.numpy()[0, 1]))
                plt.plot(x, y_t, lw=1, dashes=iter_styles.next(), color=c,
                         label=r'$\hat{{f}}_{}(x)$'.format(i))
                if i % 2 == 0:
                    x0 = x[0]
                    y0 = y_t[0]
                else:
                    x0 = x[-1]
                    y0 = y_t[-1]
                plt.text(x0, y0, str(i), size=10, color=c)
        plt.plot(x, y, 'r-', lw=3, dashes=[4, 2, 2, 4, 2, 4], label=r'$\hat{{f}}_{{}}(x)$'.format(num_points))
        plt.text(x[-1], y[-1], str(num_points), size=10, color='red')
        plt.legend(loc="best", prop={'size': 8})
        frame = plt.gca()
        # frame.axes.get_xaxis().set_ticks([])
        # frame.axes.get_yaxis().set_ticks([])
        frame.set_axis_bgcolor('lightslategray')
        frame.grid(True)
        # plt.axis('off')

        if do_save:
            dt = datetime.now(timezone('Europe/Berlin')).strftime('%Y-%m-%d %H:%M:%S.%f')[-15:-7]
            exp_label = ""
            if exper is not None:
                exp_label = create_exp_label(exper) + "_"

            if fig_name is None:
                fig_name_prefix = os.path.join(exper.output_dir, os.path.join(config.figure_path,
                                                                              str(exper.epoch) + "_f" + str(f_idx)))
                fig_name = fig_name_prefix + "_" + exp_label + dt + "_" + str(num_points) + "st.png"
            else:
                fig_name = fig_name + "_" + exp_label + dt + "_" + str(num_points) + "st.png"

            plt.savefig(fig_name) # , bbox_inches='tight')
            print("INFO - Successfully saved fig %s" % fig_name)

        if show:
            plt.show()

        plt.close()

    def plot_opt_steps(self, f_idx, fig_name=None, height=8, width=6, do_save=False, show=False, exper=None,
                       add_text=None):
        MAP = config.color_map
        cm = plt.get_cmap(MAP)
        num_points = len(self.param_hist) - 1
        f_true_desc = r'$f(x)=$' + self.poly_desc(f_idx) + r'$ (steps={})$'.format(num_points) + "\n"
        f_approx_desc = r'$\hat{f}(x)=$' + self.poly_desc(f_idx, f_true=False)
        l_title = f_true_desc + f_approx_desc
        plt.figure(figsize=(height, width))
        plt.title(l_title)
        ax = plt.gca()
        axis_ranges = self.plot_contour(f_idx)
        plt.scatter(self.true_params[f_idx, 0].data.cpu().numpy(), self.true_params[f_idx, 1].data.cpu().numpy(),
                    color="red", marker="H", s=400, alpha=0.5)
        losses = []
        if num_points > 1:
            ax.set_prop_cycle(cycler('color', [cm(1.15 * i / (num_points)) for i in range(num_points)]))
            for i in range(num_points):
                params = self.param_hist[i][f_idx, :].unsqueeze(0)
                y_t = torch.mm(self.self.W[f_idx], params).data.numpy().squeeze()
                y = self.y.data.numpy()[f_idx, :]
                loss = 0.5 * 1/self.stddev**2 * np.sum((y - y_t)**2)
                losses.append(loss)
                x_array = [self.param_hist[i][f_idx, 0].data.numpy()[0],
                           self.param_hist[i+1][f_idx, 0].data.numpy()[0]]
                y_array = [self.param_hist[i][f_idx, 1].data.numpy()[0],
                           self.param_hist[i+1][f_idx, 1].data.numpy()[0]]
                ax.plot(x_array, y_array, 'o-')
                # add the step number, although this still plots the number sometimes slightly off
                if i == num_points - 1:
                    a_color = "navajowhite"
                    x_pos = int(self.param_hist[i+1][f_idx, 0].data.numpy()[0])
                    y_pos = int(self.param_hist[i+1][f_idx, 1].data.numpy()[0])
                    plt.annotate(str(i+1), xy=(x_pos, y_pos), size=8, color=a_color)
            # compute the last loss, e.g. in case we have 10 opt steps we have 11 parameters in our history
            # because the initial one counts as t_0
            params = self.param_hist[i+1][f_idx, :].unsqueeze(0)
            y_t = torch.mm(self.W[f_idx], params).data.numpy().squeeze()
            y = self.y.data.numpy()[f_idx, :]
            loss = 0.5 * 1 / self.stddev**2 * np.sum((y - y_t) ** 2)
            losses.append(loss)
        # ax.axes.get_xaxis().set_ticks([])
        # ax.axes.get_yaxis().set_ticks([])
        # ax.set_axis_bgcolor('lightslategray')
        ax.grid(True)
        ax.set_xlim([axis_ranges[0], axis_ranges[1]])
        ax.set_ylim([axis_ranges[2], axis_ranges[3]])
        # if add_text[0] is not None and add_text[1] is not None:
        #     text = "q(t) values {}".format(add_text[0]) + "\n" + "losses {}".format(add_text[1])
        #     annotation = True
        if len(losses) != 0:
            # we end up here if we're are training the meta learner without ACT, so we only have losses
            losses = np.array_str(np.array(losses), precision=3, suppress_small=True)
            text = "losses {}".format(losses)
            annotation = True
        else:
            annotation = False

        # if annotation:
        if 1 == 0:
            plt.annotate(text,
                         xy=(0.5, 0), xytext=(0, 0),
                         xycoords=('axes fraction', 'figure fraction'),
                         textcoords='offset points',
                         size=8, ha='center', va='bottom')

        if do_save:
            dt = datetime.now(timezone('Europe/Berlin')).strftime('%Y-%m-%d %H:%M:%S.%f')[-15:-7]
            exp_label = ""
            if exper is not None:
                exp_label = create_exp_label(exper) + "_"

            if fig_name is None:
                fig_name_prefix = os.path.join(exper.output_dir, os.path.join(config.figure_path,
                                                                              str(exper.epoch) + "_f" + str(f_idx) +
                                                                              "_opt_steps"))
                fig_name = fig_name_prefix + "_" + exp_label + dt + "_" + str(num_points) + "st.png"
            else:
                fig_name = fig_name + "_" + exp_label + dt + "_" + str(num_points) + "st.png"

            plt.savefig(fig_name) # , bbox_inches='tight')
            print("INFO - Successfully saved fig %s" % fig_name)

        if show:
            plt.show()

        plt.close()
