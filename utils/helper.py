import torch


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
        grad_input = grad_output.clone()
        grad_input = torch.mul(grad_input, mask).double()
        grad_output = torch.mul(grad_output, mask).double()

        return grad_input, grad_output


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

