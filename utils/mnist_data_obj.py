import torch
from torchvision import datasets, transforms
from torch.autograd import Variable


class MNISTDataSet(object):

    def __init__(self, train_batch_size=128, test_batch_size=256, use_cuda=False):
        self.use_cuda = use_cuda
        self.batch_size = train_batch_size
        kwargs = {'num_workers': 1, 'pin_memory': True} if self.use_cuda else {}
        self.train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=self.batch_size, shuffle=True, **kwargs)

        self.test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])),
            batch_size=test_batch_size, shuffle=True, **kwargs)

        self.train_iter = iter(self.train_loader)
        self.test_iter = iter(self.test_loader)

    def next_batch(self, is_train=True):
        try:
            if not is_train:
                x, y = next(self.test_iter)
            else:
                x, y = next(self.train_iter)
        except StopIteration:
            if not is_train:
                self.test_iter = iter(self.test_loader)
                x, y = next(self.test_iter)
            else:
                self.train_iter = iter(self.train_loader)
                x, y = next(self.train_iter)

        if self.use_cuda:
            x, y = x.cuda(), y.cuda()

        return Variable(x), Variable(y)