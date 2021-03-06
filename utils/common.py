import math
import torch
import os
import sys
import dill
from config import config
from collections import OrderedDict

import argparse
import logging
import numpy as np
from scipy.stats import geom

from torch.autograd import Variable
import models.rnn_optimizer
import models.sb_act_optimizer
from probs import ConditionalTimeStepDist, TimeStepsDist
from regression import RegressionFunction, L2LQuadratic, RosenBrock, RegressionWithStudentT

from mlp import MLP


CANONICAL = False

MODEL_DICT = {'metaV1': "LSTM baseline",
              'metaV7': "LSTM curriculum",
              'meta_actV1': "M-ACT",
              'act_sbV3.2': "M-PACT"}

OPTIMIZER_DICT = {'sgd': torch.optim.SGD, # Gradient Descent
                  'adadelta': torch.optim.Adadelta, # Adadelta
                  'adagrad': torch.optim.Adagrad, # Adagrad
                  'adam': torch.optim.Adam, # Adam
                  'rmsprop': torch.optim.RMSprop # RMSprop
                  }


default_mlp_architecture = \
    dict(loss_function="CrossEntropyLoss",  # Loss function
         n_input=784,                       # MNIST data input (img shape: 28*28)
         n_hidden_layer1=20,                # number of hidden units first hidden layer
         act_function="Sigmoid",            # activation function
         n_output=10                        # MNIST number of output tokens
         )

two_layer_mlp_architecture = \
    dict(loss_function="CrossEntropyLoss",  # Loss function
         n_input=784,                       # MNIST data input (img shape: 28*28)
         n_hidden_layer1=20,                # number of hidden units first hidden layer
         n_hidden_layer2=20,                # number of hidden units 2nd hidden layer
         act_function="Sigmoid",            # activation function
         n_output=10                        # MNIST number of output tokens
         )


test40_mlp_architecture = \
    dict(loss_function="CrossEntropyLoss",  # Loss function
         n_input=784,                       # MNIST data input (img shape: 28*28)
         n_hidden_layer1=40,                # number of hidden units first hidden layer
         act_function="Sigmoid",            # activation function
         n_output=10                        # MNIST number of output tokens
         )

testRELU_mlp_architecture = \
    dict(loss_function="CrossEntropyLoss",  # Loss function
         n_input=784,                       # MNIST data input (img shape: 28*28)
         n_hidden_layer1=20,                # number of hidden units first hidden layer
         act_function="ReLU",               # activation function
         n_output=10                        # MNIST number of output tokens
         )


def compute_mean_ptx(np_ptx, halt_steps):
    N = np_ptx.shape[0]
    res = halt_steps.nonzero()  # returns tuple, we're only interesting in indices
    T = res[0]
    num_of_keys = T.shape[0]
    ptx_T = OrderedDict(zip(T, [[] for _ in range(num_of_keys)]))

    # collect the optimizee rho values with the same horizon length
    for i in np.arange(N):
        i_ptx = np_ptx[i, :]
        l_idx = np.max(i_ptx.nonzero())
        T_x = l_idx + 1  # horizon for this optimizee
        if T_x == 1:
            res = i_ptx[0]
        else:
            res = i_ptx[0: T_x]  # rhos up to the penultimate

        ptx_T[T_x].append(res)

    for H, ptxT in ptx_T.iteritems():
        ptxT = np.mean(np.array(ptxT), axis=0)
        ptxT *= 1. / np.sum(ptxT)
        ptx_T[H] = ptxT

    return ptx_T


def compute_mean_qTx_qzxT(m_rhos, halt_steps):
    # number of optimizees
    N = m_rhos.shape[0]
    res = halt_steps.nonzero()  # returns tuple, we're only interesting in indices
    T = res[0]
    num_of_keys = T.shape[0]
    rhos_T = OrderedDict(zip(T, [[] for _ in range(num_of_keys)]))
    pT_H = OrderedDict()
    pzT_H = OrderedDict()

    # collect the optimizee rho values with the same horizon length
    for i in np.arange(N):
        i_rhos = m_rhos[i, :]
        l_idx = np.max(i_rhos.nonzero())
        T_x = l_idx + 1  # horizon for this optimizee
        if T_x == 1:
            res = i_rhos[0]
        else:
            res = i_rhos[0: T_x]  # rhos up to the penultimate

        rhos_T[T_x].append(res)

    # compute the mean rhos values per horizon H
    for H, rhos in rhos_T.iteritems():
        if len(rhos) > 0:
            np_rhos = np.mean(np.array(rhos), axis=0)
            # now construct the p(T) probs for all horizons T<=H
            pT = np.zeros(H)
            pzT = np.zeros(H)
            for T in np.arange(H):
                if T == 0:
                    pT[0] = np_rhos[0]
                    pzT[0] = np_rhos[0]
                else:
                    product = np.prod(1 - np_rhos[0:T])
                    pT[T] = product * np_rhos[T]
                    if T + 1 == H:  # last time step
                        pzT[T] = 1 - np.sum(pzT)
                    else:
                        # strange but true, our pzT values are equal to the p(T) values except for the last step
                        pzT[T] = pT[T]
            # save the mean rho values per horizon
            rhos_T[H] = np_rhos
            pT_H[H] = pT
            pzT_H[H] = pzT

    return rhos_T, pT_H, pzT_H


def compute_marginal_q_z(qT_H, qzT_H):
    qz_H = OrderedDict()
    max_H = np.max(qT_H.keys())

    for T, qzT in qzT_H.iteritems():
        q_z = np.zeros(T)
        qT = qT_H[T]
        for t in np.arange(T):
            if t == 0:
                q_z[t] = qT[0] + (1 - qzT[0]) * qzT[0]
            else:
                q_z[t] = qT[t] * (1 - np.sum(qzT[0:t])) + (1 - np.sum(qT[0:t])) * qzT[t]
        qz_H[T] = q_z * 1. / np.sum(q_z)
        # tinkering...extending all arrays to the maximum horizon length
        # the underlying problem: I my q_z values are still calculated per horizon H...in the end
        # I somehow(?) need to average over these values to obtain one q_z distribution
        # still a blind spot for me to make this work
        qz_H[T] = np.lib.pad(qz_H[T], (0, (max_H - qz_H[T].shape[0])), 'constant', constant_values=0)

    qz = np.mean(np.vstack(qz_H.values()), axis=0)
    qz *= 1. / np.sum(qz)
    return qz


def get_official_model_name(exper):

    key_value = exper.args.learner+exper.args.version
    return MODEL_DICT.get(key_value, key_value)


def create_logger(exper=None, file_handler=False, output_dir=None):
    # create logger
    if exper is None and output_dir is None:
        raise ValueError("Parameter -experiment- and -output_dir- cannot be both equal to None")
    logger = logging.getLogger('meta learner')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    if file_handler:
        if output_dir is None:
            output_dir = exper.output_dir
        fh = logging.FileHandler(os.path.join(output_dir, config.logger_filename))
        # fh.setLevel(logging.INFO)
        fh.setLevel(logging.DEBUG)
        formatter_fh = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter_fh)
        logger.addHandler(fh)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers

    formatter_ch = logging.Formatter('%(levelname)s - %(message)s')
    ch.setFormatter(formatter_ch)
    # add the handlers to the logger
    logger.addHandler(ch)

    return logger


def print_flags(exper):
    """
    Prints all entries in argument parser.
    """
    for key, value in vars(exper.args).items():
        exper.meta_logger.info(key + ' : ' + str(value))

    if exper.args.learner == 'act' or (exper.args.learner == 'meta' and exper.args.version == 'V3'):
        exper.meta_logger.info("shape parameter of prior p(t|T) nu={:.3}".format(exper.config.ptT_shape_param))
    if exper.args.learner[0:6] == 'act_sb':
        exper.meta_logger.info("shape parameter of prior p(t|nu={:.3f})".format(exper.config.ptT_shape_param))
        if exper.args.version == "V2" or exper.args.version == "V3.2":
            exper.meta_logger.info(" ! NOTE: using KL cost annealing")
            exper.meta_logger.info(">>> Annealing schedule {}".format(np.array_str(exper.annealing_schedule)))

    if exper.args.learner == "meta_act":
        exper.meta_logger.info("Hyperparameter for meta-act model: tau={:.6f}".format(exper.config.tau))
    if exper.args.learner[0:3] == 'act' or (exper.args.learner == 'meta' and exper.args.version == 'V2') \
            or exper.args.learner == "meta_act":
        if not exper.args.fixed_horizon:
            exper.meta_logger.info("Maximum horizon constraint to {} due to memory "
                                   "shortage".format(exper.config.T))
    if exper.args.learner == 'meta' and exper.args.version == 'V7':
        exper.meta_logger.info("Please note that MetaLearner is trained incrementally")
        exper.meta_logger.info(np.array_str(exper.inc_learning_schedule))

    if exper.args.trunc_bptt:
        exper.meta_logger.info(" IMPORTANT >>> Using truncated BPTT {} ".format(exper.args.truncated_bptt_step))
    else:
        exper.meta_logger.info(" IMPORTANT >>> NOT USING truncated BPTT")
    exper.meta_logger.info("NOTE: >>> Using batch handler class {} <<<".format(exper.batch_handler_class))


def compute_total_steps(exper, epoch_range=None, is_train=True):
    if is_train:
        keys = exper.epoch_stats["halting_step"].keys()
        halt_dict = exper.epoch_stats["halting_step"]
    else:
        keys = exper.val_stats["halting_step"].keys()
        halt_dict = exper.val_stats["halting_step"]
    if epoch_range is None:
        epoch_range = [keys[0], keys[-1]]

    total_steps = 0
    for key, halting_steps in halt_dict.items():

        if epoch_range[0] <= key <= epoch_range[1]:
            num_of_steps = halting_steps.shape[0]
            values = np.arange(0, num_of_steps)
            total_steps += np.sum(values * halting_steps)

    return total_steps


def get_evaluation_step_loss(exper, steps=[1, 5, 50], epoch=None):
    if epoch is None:
        epoch = exper.val_stats["step_losses"].keys()[-1]

    step_losses = exper.val_stats["step_losses"][epoch]
    last_step = step_losses.shape[0] - 1
    # remember "step_losses" also incorporates the 0-step (initial loss), therefore we can easily
    # work with the real indices
    steps.extend([last_step])

    return step_losses[np.array(steps)]


def halting_step_stats(halting_steps):
    num_of_steps = halting_steps.shape[0]
    num_of_funcs = np.sum(halting_steps)
    values = np.arange(0, num_of_steps)
    total_steps = np.sum(values * halting_steps)
    avg_opt_steps = np.sum(1. / num_of_funcs * values * halting_steps)
    E_x_2 = np.sum(1. / float(num_of_funcs) * values ** 2 * halting_steps)
    stddev = np.sqrt(E_x_2 - avg_opt_steps ** 2)
    cum_sum = np.cumsum(halting_steps)
    if cum_sum[np.nonzero(cum_sum)[0][0]] > num_of_funcs / 2.:
        median = np.nonzero(cum_sum)[0][0]
    else:
        median = np.argmax(cum_sum[cum_sum < num_of_funcs / 2.]) + 1

    return avg_opt_steps, stddev, median, total_steps


def compute_mean_training_horizon(exper, last_epoch=None):
    if last_epoch is None:
        last_epoch = exper.args.max_epoch

    sum_horizons = 0
    for e in np.arange(1, last_epoch):
        halt_steps = exper.epoch_stats["halting_step"][e]
        mean_horizon, _, _, _ = halting_step_stats(halt_steps)
        sum_horizons += mean_horizon

    return sum_horizons / float(last_epoch)


def softmax(x, dim=1):
    """Compute softmax values for each sets of scores in x. Expecting numpy arrays"""
    e_x = np.exp(x - np.max(x, dim, keepdims=True))

    return e_x / e_x.sum(axis=dim, keepdims=True)


def stop_computing(q_probs, threshold=0.8):
    """
    Used during validation to determine the moment to stop computation based on qt-values

    :param q_probs: contains the qt probabilities up to horizon T: [batch_size, opt_steps]
    :return: vector of booleans [batch_size]
    """
    # remember, we call this function while we collected time-steps qt values up to T=t (current optimization step)
    # and we want to test whether
    stops = np.cumsum(q_probs, 1)[:, -2] > threshold
    return stops


def preprocess_gradients(x):
    """
    Taken from another pytorch implementation of the L2L model.
    This is the pre-processing as mentioned in appendix A of the L2L paper.
    Should solve the problem that the gradients of the different parameters of the MLP have VERY different magnitudes.

    :param x: gradients of all MLP parameters size=[num of parameters, 1]
    :return: FloatTensor size=[number of parameters, 2]
    """
    p = 10
    eps = 1e-6
    indicator = (x.abs() > math.exp(-p)).float()
    x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)

    return torch.cat((x1, x2), 1)


def create_def_argparser(**kwargs):

    args = argparse.Namespace()
    args.cuda = kwargs['cuda']
    args.model = kwargs['model']
    args.log_dir = kwargs['log_dir']
    args.learner = kwargs['learner']
    args.num_layers = 2
    args.hidden_size = 20
    args.retrain = kwargs['retrain']
    args.truncated_bptt_step = kwargs['truncated_bptt_step']
    args.lr = kwargs['lr']

    return args


def get_model(exper, num_inputs, retrain=False):

    if exper.args.learner == "act":
        # currently two versions of AML in use. V1 is now the preferred, which has the 2nd LSTM for the
        # q(t|T, theta) approximation incorporated into the first LSTM (basically one loop, where V2 has
        # 2 separate loops and works on the mean of the gradients, where V1 works on the individual parameters
        # only use first 2 characters of args.version e.g. V1 instead of V1.1
        str_classname = "AdaptiveMetaLearner" + exper.args.version[0:2]
        act_class = getattr(models.rnn_optimizer, str_classname)
        meta_optimizer = act_class(num_inputs=num_inputs,
                                   num_layers=exper.args.num_layers,
                                   num_hidden=exper.args.hidden_size,
                                   use_cuda=exper.args.cuda,
                                   output_bias=exper.args.output_bias)
    elif exper.args.learner[0:6] == "act_sb" or exper.args.learner == "meta_act":
        str_classname = "StickBreakingACTBaseModel"
        act_class = getattr(models.sb_act_optimizer, str_classname)
        meta_optimizer = act_class(num_inputs=num_inputs,
                                   num_layers=exper.args.num_layers,
                                   num_hidden=exper.args.hidden_size,
                                   use_cuda=exper.args.cuda,
                                   output_bias=exper.args.output_bias)
    elif exper.args.learner == "meta":
        # the alternative model is our MetaLearner in different favours
        if exper.args.version[0:2] == "V4":
            str_classname = "MetaStepLearner"
            meta_class = getattr(models.rnn_optimizer, str_classname)
        elif exper.args.version[0:2] == "V5":
            str_classname = "MetaLearnerWithValueFunction"
            meta_class = getattr(models.rnn_optimizer, str_classname)
        else:
            str_classname = "MetaLearner"
            meta_class = getattr(models.rnn_optimizer, str_classname)

        meta_optimizer = meta_class(num_inputs=num_inputs,
                                    num_layers=exper.args.num_layers,
                                    num_hidden=exper.args.hidden_size,
                                    use_cuda=exper.args.cuda,
                                    output_bias=exper.args.output_bias)

    if retrain:
        loaded = False
        if exper.model_path is not None:
            # try to load model
            if os.path.exists(exper.model_path):
                meta_optimizer.load_state_dict(torch.load(exper.model_path))
                exper.meta_logger.info("INFO - loaded existing model from file {}".format(exper.model_path))
                loaded = True
            else:
                exper.meta_logger.info("Warning - model not found in {}".format(exper.model_path))

        if exper.args.model is not None and not loaded:
            model_file_name = os.path.join(config.model_path, (exper.args.model + config.save_ext))
            if os.path.exists(model_file_name):
                meta_optimizer.load_state_dict(torch.load(model_file_name))
                exper.meta_logger.info("INFO - loaded existing model from file {}".format(model_file_name))
                loaded = True

        if not loaded:
            exper.meta_logger.info("Warning - retrain was enabled but model <{}> does not exist. "
                                   "Training a new model with this name.".format(exper.args.model))

    else:
        exper.meta_logger.info("INFO - training a new model {}".format(exper.args.model))
    meta_optimizer.name = exper.args.model

    if exper.args.cuda:
        exper.meta_logger.info("Note: {} is running on GPU".format(str_classname))
        meta_optimizer.cuda()
    else:
        exper.meta_logger.info("Note: {} is running on CPU".format(str_classname))
    # print(meta_optimizer.state_dict().keys())
    param_list = []
    for name, param in meta_optimizer.named_parameters():
        param_list.append(name)

    exper.meta_logger.info(param_list)

    return meta_optimizer


def get_batch_functions(exper, binary_switch=1):
    if exper.args.problem == "quadratic":
        funcs = L2LQuadratic(batch_size=exper.args.batch_size, num_dims=exper.args.x_dim, stddev=0.01,
                                 use_cuda=exper.args.cuda)

    elif exper.args.problem == "regression":
        funcs = RegressionFunction(n_funcs=exper.args.batch_size, n_samples=exper.args.x_samples,
                                   stddev=exper.config.stddev, x_dim=exper.args.x_dim,
                                   use_cuda=exper.args.cuda)
    elif exper.args.problem == "regression_T":
        funcs = RegressionWithStudentT(n_funcs=exper.args.batch_size, n_samples=exper.args.x_samples,
                                       x_dim=exper.args.x_dim, scale_p=1., shape_p=1,
                                       use_cuda=exper.args.cuda)
    elif exper.args.problem == "rosenbrock":
        funcs = RosenBrock(batch_size=exper.args.batch_size, stddev=exper.config.stddev, num_dims=2,
                           use_cuda=exper.args.cuda, canonical=CANONICAL)
    elif exper.args.problem == "mlp":
        if exper.args.mixed_mlp:
            # alternative between 1-layer and 2-layer MLP
            if exper.binary_switch == 0:
                funcs = MLP(default_mlp_architecture)
                exper.binary_switch = 1
            else:
                funcs = MLP(two_layer_mlp_architecture)
                exper.binary_switch = 0
        else:
            funcs = MLP(default_mlp_architecture)
        if exper.args.cuda:
            funcs = funcs.cuda()
    else:
        raise ValueError("Problem type >>>{}<<< is not supported".format(exper.args.problem))

    return funcs


def get_func_loss(exper, funcs, average=False, is_train=True):
    if exper.args.problem == "quadratic":
        loss = funcs.compute_loss(average=average)
    elif exper.args.problem == "regression":
        loss = funcs.compute_neg_ll(average_over_funcs=average, size_average=False)
    elif exper.args.problem == "regression_T":
        loss = funcs.compute_neg_ll(average_over_funcs=average)
    elif exper.args.problem == "rosenbrock":
        loss = funcs(average_over_funcs=average)
    elif exper.args.problem == "mlp":
        image, y_true = exper.dta_set.next_batch(is_train=is_train)
        loss = funcs.evaluate(image, compute_loss=True, y_true=y_true)

    return loss


def load_val_data(path_specs=None, num_of_funcs=10000, n_samples=100, stddev=1., dim=2, logger=None, file_name=None,
                  exper=None):

    if exper.args.problem == "rosenbrock":
        logger.info("Note CANONICAL set to {}".format(CANONICAL))
    if file_name is None:
        if exper.args.problem == "mlp":
            num_of_funcs = 5
            if not exper.args.mixed_mlp:
                file_name = config.val_file_name_suffix + exper.args.problem + "_" + str(num_of_funcs) + ".dll"
            else:
                file_name = config.val_file_name_suffix + "mixed_" + exper.args.problem + "_" \
                            + str(num_of_funcs) + ".dll"
        else:
            file_name = config.val_file_name_suffix + exper.args.problem + "_" + str(num_of_funcs) + "_" + \
                        str(n_samples) + "_" + \
                        str(stddev) + "_" + \
                        str(dim) + ".dll"

    if path_specs is not None:
        load_file = os.path.join(path_specs, file_name)
    else:
        load_file = os.path.join(config.data_path, file_name)
    try:
        if os.path.exists(load_file):

            with open(load_file, 'rb') as f:
                val_funcs = dill.load(f)
            logger.info("INFO - validation set loaded from {}".format(load_file))
        else:
            if exper.args.problem == "regression":
                val_funcs = RegressionFunction(n_funcs=num_of_funcs, n_samples=exper.args.x_samples, stddev=stddev,
                                               x_dim=exper.args.x_dim, use_cuda=exper.args.cuda,
                                               calc_true_params=False)
            elif exper.args.problem == "regression_T":
                val_funcs = RegressionWithStudentT(n_funcs=num_of_funcs, n_samples=exper.args.x_samples,
                                                   x_dim=exper.args.x_dim, scale_p=1., shape_p=1,
                                                   use_cuda=exper.args.cuda)
            elif exper.args.problem == "quadratic":
                val_funcs = L2LQuadratic(batch_size=num_of_funcs, num_dims=exper.args.x_dim,
                                                     stddev=stddev, use_cuda=exper.args.cuda)
            elif exper.args.problem == "rosenbrock":
                val_funcs = RosenBrock(batch_size=num_of_funcs, stddev=stddev, num_dims=exper.args.x_dim,
                                       use_cuda=exper.args.cuda, canonical=CANONICAL)

            elif exper.args.problem == "mlp":
                val_funcs = []
                num_of_funcs = 5
                binary_switch = 0
                for _ in np.arange(num_of_funcs):
                    if not exper.args.mixed_mlp:
                        val_funcs.append(MLP(default_mlp_architecture))
                    else:
                        if binary_switch == 0:
                            val_funcs.append(MLP(default_mlp_architecture))
                            binary_switch = 1
                        else:
                            val_funcs.append(MLP(two_layer_mlp_architecture))
                            binary_switch = 0
            else:
                raise ValueError("Problem type {} is not supported".format(exper.args.problem))
            logger.info("Creating validation set of size {} for problem {}".format(num_of_funcs, exper.args.problem))
            with open(load_file, 'wb') as f:
                dill.dump(val_funcs, f)
            logger.info("Successfully saved validation file {}".format(load_file))

    except:
        raise IOError("Can't open file {}".format(load_file))

    return val_funcs


def detailed_train_info(logger, func, f_idx, step, optimizer_steps, error):
    logger.info("INFO-track -----------------------------------------------------")
    logger.info("{}-th batch (op-steps {}): loss {:.4f}".format(step, optimizer_steps, error))
    logger.info("Example function: {}".format(func.poly_desc(f_idx)))
    p_init = func.initial_params[f_idx, :].data.numpy().squeeze()
    logger.info("Initial parameter values {}".format(np.array_str(p_init, precision=3)))
    p_true = func.true_params[f_idx, :].data.numpy().squeeze()
    logger.info("True parameter values {}".format(np.array_str(p_true, precision=3)))
    params = func.params[f_idx, :].data.numpy().squeeze()
    logger.info("Final parameter values {})".format(np.array_str(params, precision=3)))


def construct_prior_p_t_T(optimizer_steps, continue_prob, batch_size, cuda=False):

    prior_dist = ConditionalTimeStepDist(T=optimizer_steps, q_prob=continue_prob)
    # The range that we pass to pmfunc (to compute the priors of p(t|T)) ranges from 1...T
    # because we define t as the "trial number of the first success"!
    prior_probs = Variable(torch.from_numpy(prior_dist.pmfunc(T=optimizer_steps, normalize=True)).float())
    if cuda:
        prior_probs = prior_probs.cuda()
    # we need to expand the prior probs to the size of the batch
    return prior_probs.expand(batch_size, prior_probs.size(0))


def generate_fixed_weights(exper, steps=None):

    if steps is None:
        steps = exper.args.optimizer_steps

    fixed_weights = None
    if exper.args.learner == 'meta' and (exper.args.version[0:2] == "V3" or exper.args.version[0:2] == "V5"):
        # Version 3.1 of MetaLearner uses a fixed geometric distribution as loss weights
        if exper.args.version == "V3.1":
            exper.meta_logger.info("Model with fixed weights from geometric distribution p(t|{},{:.3f})".format(
                steps, exper.config.ptT_shape_param))
            prior_probs = geom.pmf(np.arange(1, steps+1), p=(1-exper.config.ptT_shape_param))
            prior_probs = 1. / np.sum(prior_probs) * prior_probs
            prior_probs = np.array([0.607,   0.3749,  0.1777,  0.0779,  0.0288,  0.0215,  0.0145])
            prior_probs = Variable(torch.from_numpy(prior_probs).float())
            if exper.args.cuda:
                prior_probs = prior_probs.cuda()

            fixed_weights = prior_probs.squeeze()
            exper.meta_logger.info(fixed_weights)

        elif exper.args.version == "V3.2":
            fixed_weights = Variable(torch.FloatTensor(steps), requires_grad=False)
            fixed_weights[:] = 1. / float(steps)
            exper.meta_logger.info("Model with fixed uniform weights that sum to {:.1f}".format(
                torch.sum(fixed_weights).data.cpu().squeeze()[0]))
        elif exper.args.version[0:2] == "V5":
            # in metaV5 we construct the loss-weights based on the RL approach for cumulative discounted reward
            # we take gamma = ptT_shape_param - that we normally use to construct the prior geometric distribution
            weights = [Variable(torch.FloatTensor([exper.config.ptT_shape_param**i]))
                       for i in np.arange(steps)]
            fixed_weights = torch.cat(weights)
    else:
        fixed_weights = Variable(torch.ones(steps))

    if exper.args.cuda and not fixed_weights.is_cuda:
        fixed_weights = fixed_weights.cuda()

    return fixed_weights


def transform_halting_steps_to_opt_steps(halt_steps, last_epoch=None):
    r = []
    for i, steps in enumerate(halt_steps):
        max_idx = np.max(steps.nonzero())
        # skip first 0-step
        l = []
        for j, counter in enumerate(steps[1:max_idx+1]):
            if counter != 0:
                l.extend([j+1] * counter)
        r.append(l)
    return r


def load_curriculum(file_name):
    filepath = os.path.join("data", file_name)
    try:
        with open(filepath, 'rb') as f:
            schedule = dill.load(f)
    except:
        raise IOError("Can't open file {}".format(filepath))

    return schedule