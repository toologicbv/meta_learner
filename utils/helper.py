import torch
from config import config
import math


from regression import neg_log_likelihood_loss, nll_with_t_dist
from regression import RegressionFunction, RegressionWithStudentT
from utils.mlp import MLP


class LessOrEqual(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    def forward(self, input1, input2):
        """
        In the forward pass we receive a Tensor containing the input and return a
        Tensor containing the output. You can cache arbitrary Tensors for use in the
        backward pass using the save_for_backward method.
        input1: the new cumulative probabilities [batch_size, 1]
        input2: 1-epsilon, the threshold.
        Note:
            so we are generating a ByteTensor mask here.
            0 = the sum of the probs (for this optimizee) reached the threshold, t = N(t) in Graves paper
            1 = cumulative probs are still less than 1-epsilon continue optimizing
        """
        mask = torch.le(input1, input2).float()
        self.save_for_backward(mask)
        return mask

    def backward(self, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        mask, = self.saved_tensors
        grad_input2 = None
        grad_input1 = grad_output.clone()
        grad_input1[:] = config.tau
        grad_input1 = torch.mul(grad_input1, -mask).double()

        return grad_input1, grad_input2


def normalize(v):
    """
    assuming a python tensor with one dimension
    :param v:
    :return: a normalized vector v
    """
    factor = (1./torch.sum(v)).data.cpu().squeeze().numpy()[0]
    return v * float(factor)


def tensor_and(t1, t2):
    res = (t1+t2).eq(2)
    if t1.is_cuda:
        res = res.cuda()
    return res


def tensor_any(t1):
    res = torch.sum(t1).gt(0)
    if t1.is_cuda:
        res = res.cuda()
    return res


def preprocess_gradients(x):
    p = 10
    eps = 1e-6
    indicator = (x.abs() > math.exp(-p)).float()
    x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)

    return torch.cat((x1, x2), 1)


def get_step_loss(optimizee_obj, new_parameters, avg_batch=False, exper=None, is_train=True):
    if optimizee_obj.__class__ == RegressionFunction:
        loss = neg_log_likelihood_loss(optimizee_obj.y, optimizee_obj.y_t(new_parameters),
                                       stddev=optimizee_obj.stddev, N=optimizee_obj.n_samples,
                                       avg_batch=avg_batch, size_average=False)
    elif optimizee_obj.__class__ == RegressionWithStudentT:
        loss = nll_with_t_dist(optimizee_obj.y, optimizee_obj.y_t(new_parameters), N=optimizee_obj.n_samples,
                               shape_p=optimizee_obj.shape_p, scale_p=optimizee_obj.scale_p,
                               avg_batch=avg_batch)

    elif optimizee_obj.__class__ == MLP:
        optimizee_obj.set_eval_obj_parameters(new_parameters)
        image, y_true = exper.dta_set.next_batch(is_train=True)
        loss = optimizee_obj.evaluate(image, use_copy_obj=True, compute_loss=True, y_true=y_true, is_train=is_train)
    else:
        raise ValueError("Optimizee class not supported {}".format(optimizee_obj.__class__))

    return loss

