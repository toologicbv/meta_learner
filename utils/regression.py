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


def construct_poly_base(x, degree=2):
    x_numpy = x.data.numpy()
    x_array = []
    for i in range(1, degree + 1):
        x_array.append(x_numpy[:, np.newaxis] ** i)

    X = np.concatenate(x_array, axis=1).T
    X = Variable(torch.from_numpy(X).float())
    return X


def construct_linear_base(x):
    x_numpy = x.data.numpy()
    x_array = []
    x_array.append(np.ones((x.size(0), 1)))
    x_array.append(x_numpy[:, np.newaxis])

    X = np.concatenate(x_array, axis=1).T
    X = Variable(torch.from_numpy(X).float())
    return X


def get_f_values(X, params):
    Y = torch.mm(params, X)
    return Y


class RegressionFunction(object):

    def __init__(self, n_funcs=100, n_samples=100, param_sigma=4., noise_sigma=1.,
                 x_min=-2, x_max=2, poly_degree=2, use_cuda=False, non_linear=False):
        self.use_cuda = use_cuda # TODO not implemented yet
        self.noise_sigma = noise_sigma
        self.num_of_funcs = n_funcs
        self.n_samples = n_samples
        self.degree = poly_degree
        self.non_linear = non_linear
        self.true_params = get_true_params(size=(n_funcs, poly_degree),
                                           scale=param_sigma)
        tensor_params = init_params(self.true_params.size())
        self.params = Variable(tensor_params, requires_grad=True)
        self.initial_params = self.params.clone()

        self.x = Variable(torch.linspace(x_min, x_max, n_samples))
        if non_linear:
            self.xp = construct_poly_base(self.x, degree=poly_degree)
        else:
            self.xp = construct_linear_base(self.x)
        self.mean_values = get_f_values(self.xp, self.true_params)
        self.noise = get_noise(size=self.mean_values.size(1), sigma=noise_sigma)
        self.y = add_noise(self.mean_values, noise=self.noise)
        self.param_hist = {}
        self._add_param_values(self.initial_params)

    def enable_cuda(self):
        # TODO not implemented yet
        raise NotImplementedError("Enable cuda is not yet implemented!")

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
        self.params = Variable(reset_params, requires_grad=True)
        self._add_param_values(self.initial_params)

    def reset_params(self):
        # break the backward chain by declaring param Variable again with same Tensor values
        self.params = Variable(self.params.data, requires_grad=True)

    def compute_loss(self, average=True):
        """
        compute regression loss and if applicable apply average over functions
        :param average:
        :return: loss (Variable)
        """
        if average:
            N = float(self.num_of_funcs)
        else:
            N = 1.
        loss = 1/N * 0.5 * 1/self.noise_sigma * torch.sum((self.y - self.y_t)**2, 1)
        return loss

    def compute_neg_ll(self, average_over_funcs=False):
        ll = 0.5 * 1 / self.noise_sigma * torch.sum((self.y - self.y_t)**2, 1) + \
             self.n_samples / 2 * (np.log(self.noise_sigma) + np.log(2 * np.pi))
        if average_over_funcs:
            ll = torch.mean(ll, 0).squeeze()
        return ll

    @staticmethod
    def compute_loss_1func(sigma, y, y_hat):
        N = y.size(1)
        loss = 1 / float(N) * 0.5 * 1 / sigma * torch.sum((y - y_hat) ** 2, 1)
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
        param_error = 1/N * 0.5 * torch.sum((self.true_params - self.params) ** 2)
        return param_error.squeeze()

    @property
    def y_t(self):
        return torch.mm(self.params, self.xp)

    def poly_desc(self, idx, f_true=True):
        if f_true:
            f_params = self.true_params.data.numpy()[idx, :]
        else:
            f_params = self.params.data.numpy()[idx, :]
        descr = r'${:.3} $'.format(f_params[0])
        for i in range(1, self.degree):
            if self.non_linear:
                descr += r'$+ {:.3} x^{}$ '.format(f_params[i], i+1)
            else:
                descr += r'$+ {:.3} x$ '.format(f_params[i])
        return descr

    def get_y_values(self, params):
        return torch.mm(params, self.xp)

    def plot_contour(self, f_idx=0, delta=2):
        np_init_params = self.initial_params[f_idx, :].data.numpy()
        torch_params = self.true_params[f_idx, :].unsqueeze(0)
        np_params = self.true_params[f_idx, :].data.numpy()
        y_true = get_f_values(self.xp, torch_params)
        steps = (delta + delta) * 25
        x_range = get_axis_ranges(np_init_params[0], np_params[0], delta=delta, steps=steps)
        y_range = get_axis_ranges(np_init_params[1], np_params[1], delta=delta, steps=steps)
        X, Y = np.meshgrid(x_range, y_range)
        f_in = Variable(torch.FloatTensor(torch.cat((torch.from_numpy(X.ravel()).float(),
                                                     torch.from_numpy(Y.ravel()).float()), 1)))
        Z_mean = get_f_values(self.xp, f_in)
        Y_true = y_true.expand_as(Z_mean)
        Z = add_noise(Z_mean, noise=self.noise)

        error = 0.5 * 1 / self.noise_sigma * torch.sum((Z - Y_true) ** 2, 1)

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
        y_mean = self.mean_values.data.numpy()[f_idx, :]
        y_samples = self.y.data.numpy()[f_idx, :]
        y = self.y_t.data.numpy()[f_idx, :]
        f_true_desc = r'$f(x)=$' + self.poly_desc(f_idx) + r'$ (steps={})$'.format(num_points) + "\n"
        f_approx_desc = r'$\hat{f}(x)=$' + self.poly_desc(f_idx, f_true=False)
        l_title = f_true_desc + f_approx_desc
        plt.figure(figsize=(height, width))
        plt.title(l_title)

        plt.plot(x, y_mean, 'b-', lw=2, label=r'$f(x)$')
        plt.plot(x, y_samples, 'go', markersize=2, label=r'samples $\sigma={:.2}$'.format(self.noise_sigma))
        if steps is not None:
            for i in steps:
                p_t = self.param_hist[i][f_idx, :].unsqueeze(0)
                y_t = torch.mm(p_t, self.xp).data.numpy().squeeze()
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
                y_t = torch.mm(params, self.xp).data.numpy().squeeze()
                y = self.y.data.numpy()[f_idx, :]
                loss = 0.5 * 1/self.noise_sigma * np.sum((y - y_t)**2)
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
            y_t = torch.mm(params, self.xp).data.numpy().squeeze()
            y = self.y.data.numpy()[f_idx, :]
            loss = 0.5 * 1 / self.noise_sigma * np.sum((y - y_t) ** 2)
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

        if annotation:
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
