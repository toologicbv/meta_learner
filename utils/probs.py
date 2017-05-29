import numpy as np


def find_nearest(array, value):
    idx = (np.abs(array-value)).argmin()
    return idx, array[idx]


class TimeStepsDist(object):
    """
    Discrete probability distribution for a random variable T (time) which specifies the number of time-steps
    used for function optimization

    Method rvs:
    -----------
    draws samples from a one dimensional probability distribution,
    by means of inversion of a discrete cumulative density function

    the pdf can be sorted first to prevent numerical error in the cumulative sum
    this is set as default; for big density functions with high contrast,
    it is absolutely necessary, and for small density functions,
    the overhead is minimal

    a call to this distribution object returns indices into density array
    """
    def __init__(self, T=100, q_prob=0.9, transform=lambda x: x):
        # construct the range of possible discrete values. T denotes the absolute horizon aka support for this
        # distribution
        t_range = np.arange(1, T+1)
        self.T = T
        self.q_prob = q_prob
        pdf = self.pmfunc(t_range, normalize=True)
        self.shape = pdf.shape
        self.pdf = pdf.ravel()
        self.transform = transform
        # a pdf can not be negative
        assert(np.all(self.pdf >= 0))
        # sort the pdf by magnitude

        # construct the cumulative distribution function
        self.cdf = np.cumsum(self.pdf)
        self.probs = None

    def pmfunc(self, t, normalize=False):
        """
            Probability mass function for random variable T (discrete time), finite number of time steps
             for  a mixture of finite-time MDPs (Markov-Decision processes)
            :param t: horizon, max number of time-steps before we stop with computation
            :param prob: the probability to continue in the next time-step with the computation
            :return:
            """
        if normalize:
            assert self.T == len(t)

        pmf = self.q_prob ** (t-1) * (1. - self.q_prob ** t) * (1. - self.q_prob ** 2)
        if np.sum(pmf) < 1. and normalize and self.T > 1:
            pmf = 1./np.sum(pmf) * pmf
        return pmf

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def sum(self):
        """cached sum of all pdf values; the pdf need not sum to one, and is implicitly normalized"""
        return self.cdf[-1]

    @property
    def mean(self):
        """Markov integral of expected value of distribution"""
        return round(np.mean(self.rvs(n=100000)))

    def rvs(self, n=1, probs=False):
        """random variates """
        # pick numbers which are uniformly random over the cumulative distribution function
        choice = np.random.uniform(high=self.sum, size=n)
        # find the indices corresponding to this point on the CDF
        index = np.searchsorted(self.cdf, choice)
        self.probs = self.pdf[index]
        # because index numbering starts at 0 and we are returning the values for the random variable
        # "trial number of the first success (Bernoulli experiment)" we increment all indices by 1
        index += 1
        # is this a discrete or piecewise continuous distribution?
        if probs:
            return self.probs, self.transform(index)
        else:
            return self.transform(index)


class ConditionalTimeStepDist(object):
    """
    Discrete conditional probability distribution for a random variable t (time) conditioned on T (horizon)
    which specifies the number
    of time-steps used for function optimization

    """
    def __init__(self, T=100, q_prob=0.9, interpolation=False, transform=lambda x: x):
        # construct the range of possible discrete values. T denotes the absolute horizon aka support for this
        # distribution

        t_range = np.arange(1, T+1)
        self.T = T
        self.q_prob = q_prob
        pdf = self.pmfunc(t_range, normalize=True)
        self.shape = pdf.shape
        self.pdf = pdf.ravel()
        self.interpolation = interpolation
        self.transform = transform
        # a pdf can not be negative
        assert(np.all(self.pdf >= 0))

        # construct the cumulative distribution function
        self.cdf = np.cumsum(self.pdf)
        self.probs = None

    def pmfunc(self, t, normalize=False):
        """
        :param t:
        :param T:
        :param prob: the probability to continue in the next time-step with the computation
        :return:
        """
        if normalize:
            assert self.T == len(t)
        p = 1 - self.q_prob
        pmf = (p * self.q_prob**(t-1)) / (1 - self.q_prob**(self.T))
        if np.sum(pmf) < 1. and normalize and self.T > 1:
            pmf = 1./np.sum(pmf) * pmf
        return pmf

    @property
    def mean(self):
        return round(np.mean(self.rvs(n=100000)))

    @property
    def sum(self):
        """cached sum of all pdf values; the pdf need not sum to one, and is implicitly normalized"""
        return self.cdf[-1]

    def rvs(self, n=1, probs=False):
        """random variates """
        # pick numbers which are uniformly random over the cumulative distribution function
        choice = np.random.uniform(high=self.sum, size=n)
        # find the indices corresponding to this point on the CDF
        index = np.searchsorted(self.cdf, choice)
        self.probs = self.pdf[index]
        # because index numbering starts at 0 and we are returning the values for the random variable
        # "trial number of the first success (Bernoulli experiment)" we increment all indices by 1
        index += 1
        # is this a discrete or piecewise continuous distribution?
        if self.interpolation:
            index += + np.random.uniform(size=index.shape)
        if probs:
            return self.probs, self.transform(index)
        else:
            return self.transform(index)
