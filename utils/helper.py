import torch


def normalize(v):
    """
    assuming a python tensor with one dimension
    :param v:
    :return: a normalized vector v
    """
    factor = (1./torch.sum(v)).data.cpu().squeeze().numpy()[0]
    return v * float(factor)
