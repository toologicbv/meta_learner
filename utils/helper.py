import torch


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

