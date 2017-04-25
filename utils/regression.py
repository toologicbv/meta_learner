import numpy as np
from torch.autograd import Variable
import torch
from torch.nn import init


def get_true_params(loc=0., scale=4, size=(2, 1)):
    np_params = np.random.normal(loc=loc, scale=scale, size=size)
    return Variable(torch.FloatTensor(torch.from_numpy(np_params).float()), requires_grad=False)


def init_params(size=(1, 1)):
    theta = torch.FloatTensor(*size)
    init.uniform(theta, -0.1, 0.1)
    return Variable(theta, requires_grad=True)


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


def construct_poly_x(x, degree=2):
    x_numpy = x.data.numpy()
    x_array = []
    for i in range(1, degree + 1):
        x_array.append(x_numpy[:, np.newaxis] ** i)

    X = np.concatenate(x_array, axis=1).T
    X = Variable(torch.from_numpy(X).float())
    return X


def get_f_values(X, params):
    Y = torch.mm(params, X)
    return Y


class RegressionFunction(object):

    def __init__(self, n_funcs=100, n_samples=100, param_sigma=4., noise_sigma=1.,
                 x_min=-2, x_max=2, poly_degree=2, use_cuda=False):
        self.use_cuda = use_cuda # TODO not implemented yet
        self.noise_sigma = noise_sigma
        self.num_of_funcs = n_funcs
        self.n_samples = n_samples
        self.degree = poly_degree
        self.true_params = get_true_params(size=(n_funcs, poly_degree),
                                           scale=param_sigma)

        self.params = init_params(self.true_params.size())
        self.x = Variable(torch.linspace(x_min, x_max, n_samples))
        self.xp = construct_poly_x(self.x, degree=poly_degree)
        self.mean_values = get_f_values(self.xp, self.true_params)
        self.noise = get_noise(size=self.mean_values.size(1), sigma=noise_sigma)
        self.y = add_noise(self.mean_values, noise=self.noise)

    def set_parameters(self, parameters):
        self.params.data.copy_(parameters.data)

    def copy_params_to(self, reg_func_obj):
        # copy parameter from the meta_model function we just updated in meta_update to the
        # function that delivered the gradients...call it the "outside" function
        reg_func_obj.params.data.copy_(self.params.data)

    def reset_params(self):
        # break the backward chain by declaring param Variable again with same Tensor values
        self.params = Variable(self.params.data, requires_grad=True)

    def compute_loss(self, average=True):
        if average:
            N = float(self.n_samples)
        else:
            N = 1.
        loss = 1/N * 0.5 * 1/self.noise_sigma * torch.sum((self.y - self.y_t)**2, 1)
        return loss

    def param_error(self, average=True):
        if average:
            N = float(self.num_of_funcs)
        else:
            N = 1
        param_error = 1/N * 0.5 * torch.sum((self.true_params - self.params) ** 2)
        return param_error.squeeze()

    @property
    def y_t(self):
        return torch.mm(self.params, self.xp)

    def get_y_values(self, params):
        return torch.mm(params, self.xp)

