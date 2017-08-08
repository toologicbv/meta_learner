import math
import torch
import os
import dill
import shutil
import time
from config import config, MetaConfig
from datetime import datetime
from pytz import timezone

import argparse
import logging
import numpy as np
from collections import OrderedDict

from torch.autograd import Variable
import models.rnn_optimizer
import models.sb_act_optimizer
from plots import loss_plot, param_error_plot, plot_dist_optimization_steps, plot_qt_probs, create_exper_label
from plots import plot_image_training_loss
from probs import ConditionalTimeStepDist, TimeStepsDist
from regression import RegressionFunction, L2LQuadratic, RosenBrock, RegressionWithStudentT

CANONICAL = False

OPTIMIZER_DICT = {'sgd': torch.optim.SGD, # Gradient Descent
                  'adadelta': torch.optim.Adadelta, # Adadelta
                  'adagrad': torch.optim.Adagrad, # Adagrad
                  'adam': torch.optim.Adam, # Adam
                  'rmsprop': torch.optim.RMSprop # RMSprop
                  }


def create_logger(exper, file_handler=False):
    # create logger
    logger = logging.getLogger('meta learner')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    if file_handler:
        fh = logging.FileHandler(os.path.join(exper.output_dir, config.logger_filename))
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
    if exper.args.learner == 'act' or (exper.args.learner == 'meta' and exper.args.version == 'V2'):
        if not exper.args.fixed_horizon:
            exper.meta_logger.info("horizon limit for p(T|nu={:.3f}) due to memory "
                                   "shortage {}".format(exper.config.pT_shape_param, exper.config.T))


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
    p = 10
    eps = 1e-6
    indicator = (x.abs() > math.exp(-p)).float()
    x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)

    return torch.cat((x1, x2), 1)


def prepare_retrain(exper_path, new_args):

    exper = get_experiment(exper_path)
    exper.past_epochs = exper.args.max_epoch
    exper.args.max_epoch += new_args.max_epoch
    exper.args.lr = new_args.lr
    exper.args.eval_freq = new_args.eval_freq
    exper.args.retrain = new_args.retrain
    exper.args.log_dir = new_args.log_dir
    exper.avg_num_opt_steps = new_args.avg_num_opt_steps

    # get our logger (one to std-out and one to file)
    meta_logger = create_logger(exper, file_handler=True)

    return exper, meta_logger


def get_experiment(path_to_exp, full_path=False):

    if not full_path:
        path_to_exp = os.path.join(config.log_root_path, os.path.join(path_to_exp, config.exp_file_name))
    else:
        path_to_exp = os.path.join(config.log_root_path, path_to_exp)

    try:
        with open(path_to_exp, 'rb') as f:
            experiment = dill.load(f)
    except:
        raise IOError("Can't open file {}".format(path_to_exp))
    # needed to add this for backward compatibility, because added these parameter later
    if not hasattr(experiment.config, 'pT_shape_param'):
        new_config = MetaConfig()
        new_config.__dict__ = experiment.config.__dict__.copy()
        new_config.pT_shape_param = new_config.continue_prob
        new_config.ptT_shape_param = new_config.continue_prob
        experiment.config = new_config
    # if not hasattr(experiment.args.config, 'ptT_shape_param'):
    return experiment


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


def get_model(exper, num_params_optimizee, retrain=False, logger=None):

    if exper.args.model == "default":
        exper.args.model = exper.args.learner + exper.args.version + "_" + exper.args.problem + "_" + \
            str(int(exper.avg_num_opt_steps)) + "ops"

    if exper.args.version == 'V1' or exper.args.version == 'V2' \
        or (exper.args.version[0:2] == 'V3' and exper.args.learner == 'meta') \
            or (exper.args.version[0:2] == 'V4' and exper.args.learner == 'meta') \
            or (exper.args.version[0:2] == 'V5' and exper.args.learner == 'meta') \
            or (exper.args.version[0:2] == 'V6' and exper.args.learner == 'meta') \
            or exper.args.version == '':
        if hasattr(exper.args, 'output_bias'):
            if exper.args.output_bias:
                output_bias = True
            else:
                output_bias = False
        else:
            output_bias = True

    else:
        raise ValueError("{} version is currently not supported".format(exper.args.version))

    if exper.args.learner == "act":
        # currently two versions of AML in use. V1 is now the preferred, which has the 2nd LSTM for the
        # q(t|T, theta) approximation incorporated into the first LSTM (basically one loop, where V2 has
        # 2 separate loops and works on the mean of the gradients, where V1 works on the individual parameters
        # only use first 2 characters of args.version e.g. V1 instead of V1.1
        str_classname = "AdaptiveMetaLearner" + exper.args.version[0:2]
        act_class = getattr(models.rnn_optimizer, str_classname)
        meta_optimizer = act_class(num_params_optimizee,
                                   num_layers=exper.args.num_layers,
                                   num_hidden=exper.args.hidden_size,
                                   use_cuda=exper.args.cuda,
                                   output_bias=output_bias)
    elif exper.args.learner == "act_sb":
        str_classname = "StickBreakingACTBaseModel"
        act_class = getattr(models.sb_act_optimizer, str_classname)
        meta_optimizer = act_class(num_layers=exper.args.num_layers,
                                   num_hidden=exper.args.hidden_size,
                                   use_cuda=exper.args.cuda,
                                   output_bias=output_bias)
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

        meta_optimizer = meta_class(num_params_optimizee,
                                     num_layers=exper.args.num_layers,
                                     num_hidden=exper.args.hidden_size,
                                     use_cuda=exper.args.cuda,
                                     output_bias=output_bias)

    if retrain:
        loaded = False
        if exper.model_path is not None:
            # try to load model
            if os.path.exists(exper.model_path):
                meta_optimizer.load_state_dict(torch.load(exper.model_path))
                logger.info("INFO - loaded existing model from file {}".format(exper.model_path))
                loaded = True
            else:
                logger.info("Warning - model not found in {}".format(exper.model_path))

        if exper.args.model is not None and not loaded:
            model_file_name = os.path.join(config.model_path, (exper.args.model + config.save_ext))
            if os.path.exists(model_file_name):
                meta_optimizer.load_state_dict(torch.load(model_file_name))
                logger.info("INFO - loaded existing model from file {}".format(model_file_name))
                loaded = True

        if not loaded:
            logger.info("Warning - retrain was enabled but model <{}> does not exist. "
                        "Training a new model with this name.".format(exper.args.model))

    else:
        logger.info("INFO - training a new model {}".format(exper.args.model))
    meta_optimizer.name = exper.args.model

    if exper.args.cuda:
        logger.info("Note: {} is running on GPU".format(str_classname))
        meta_optimizer.cuda()
    else:
        logger.info("Note: {} is running on CPU".format(str_classname))
    # print(meta_optimizer.state_dict().keys())
    param_list = []
    for name, param in meta_optimizer.named_parameters():
        param_list.append(name)

    logger.info(param_list)

    return meta_optimizer


def get_batch_functions(exper):
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

    return funcs


def get_func_loss(exper, funcs, average=False):
    if exper.args.problem == "quadratic":
        loss = funcs.compute_loss(average=average)
    elif exper.args.problem == "regression":
        loss = funcs.compute_neg_ll(average_over_funcs=average, size_average=False)
    elif exper.args.problem == "regression_T":
        loss = funcs.compute_neg_ll(average_over_funcs=average)
    elif exper.args.problem == "rosenbrock":
        loss = funcs(average_over_funcs=average)

    return loss


def load_val_data(path_specs=None, num_of_funcs=10000, n_samples=100, stddev=1., dim=2, logger=None, file_name=None,
                  exper=None):

    if exper.args.problem == "rosenbrock":
        logger.info("Note CANONICAL set to {}".format(CANONICAL))
    if file_name is None:
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
            else:
                raise ValueError("Problem type {} is not supported".format(exper.args.problem))
            logger.info("Creating validation set of size {} for problem {}".format(num_of_funcs, exper.args.problem))
            with open(load_file, 'wb') as f:
                dill.dump(val_funcs, f)
            logger.info("Successfully saved validation file {}".format(load_file))

    except:
        raise IOError("Can't open file {}".format(load_file))

    return val_funcs


class Experiment(object):

    def __init__(self, run_args, config=None):
        self.args = run_args
        self.epoch_stats = {"loss": [], "param_error": [], "act_loss": [], "qt_hist": {}, "opt_step_hist": {},
                            "opt_loss": [], "step_losses": OrderedDict(), "halting_step": {}}
        self.val_stats = {"loss": [], "param_error": [], "act_loss": [], "qt_hist": {}, "opt_step_hist": {},
                          "step_losses": OrderedDict(), "opt_loss": [],
                          "step_param_losses": OrderedDict(),
                          "ll_loss": {}, "kl_div": {}, "kl_entropy": {}, "qt_funcs": OrderedDict(),
                          "loss_funcs": [], "halting_step": {}}
        self.epoch = 0
        self.output_dir = None
        self.model_path = None
        # self.opt_step_hist = None
        self.avg_num_opt_steps = 0
        self.val_avg_num_opt_steps = 0
        self.config = config
        self.val_funcs = None
        self.past_epochs = 0
        self.max_time_steps = self.args.optimizer_steps
        self.run_validation = True if self.args.eval_freq != 0 else False
        self.pt_dist = TimeStepsDist(T=self.config.T, q_prob=self.config.pT_shape_param)
        self.meta_logger = None
        self.fixed_weights = None

    def reset_val_stats(self):
        self.val_stats = {"loss": [], "param_error": [], "act_loss": [], "qt_hist": {}, "opt_step_hist": {},
                          "step_losses": OrderedDict(),
                          "step_param_losses": OrderedDict(), "ll_loss": {}, "kl_div": {}, "kl_entropy": {},
                          "qt_funcs": OrderedDict(), "loss_funcs": [], "opt_loss": []}

    def add_halting_steps(self, halting_steps):
        # expect opt_steps to be an autograd.Variable
        np_array = halting_steps.data.cpu().squeeze().numpy()
        idx_sort = np.argsort(np_array)
        np_array = np_array[idx_sort]
        vals, _, count = np.unique(np_array, return_counts=True, return_index=True)
        self.epoch_stats["halting_step"][self.epoch][vals.astype(int)] += count.astype(int)

    def add_opt_steps(self, step):
        # NOTE: we assume step starts with index 0!!! Because of numpy indexing but it is the first time step!!!
        self.epoch_stats["opt_step_hist"][self.epoch][step] += 1

    def add_step_loss(self, step_loss, step):
        # NOTE: we assume step starts with index 0!!! Because of numpy indexing but it is the first time step!!!
        if not isinstance(step_loss, (np.float, np.float32, np.float64)):
            raise ValueError("step_loss must be a numpy.float but is type {}".format(type(step_loss)))
        self.epoch_stats["step_losses"][self.epoch][step] += step_loss

    def scale_step_losses(self):
        step_loss_factors = np.zeros(self.epoch_stats["opt_step_hist"][self.epoch].shape[0])
        idx_where = np.where(self.epoch_stats["opt_step_hist"][self.epoch] > 0)
        step_loss_factors[idx_where] = 1. / self.epoch_stats["opt_step_hist"][self.epoch][idx_where]
        self.epoch_stats["step_losses"][self.epoch] *= step_loss_factors

    def start(self, epoch_obj):

        if self.args.learner == "act":
            if self.args.version[0:2] not in ['V1', 'V2']:
                raise ValueError("Version {} currently not supported (only V1.x and V2.x)".format(self.args.version))

            if self.args.fixed_horizon:
                self.avg_num_opt_steps = self.args.optimizer_steps

            else:
                self.avg_num_opt_steps = self.pt_dist.mean
                self.max_time_steps = self.config.T
        else:
            self.avg_num_opt_steps = self.args.optimizer_steps
            if self.args.learner == 'meta' and self.args.version[0:2] == 'V2':
                # Note, we choose here an absolute limit of the horizon, set in the config-file
                self.max_time_steps = self.config.T
                self.avg_num_opt_steps = self.pt_dist.mean
            elif self.args.learner == 'meta' and (self.args.version[0:2] == 'V5' or self.args.version[0:2] == 'V6'):
                # disable BPTT by setting truncated bptt steps to optimizer steps
                self.args.truncated_bptt_step = self.args.optimizer_steps

        if self.args.problem == "rosenbrock":
            assert self.args.learner == "meta", "Rosenbrock problem is only suited for MetaLearner"
            self.config.max_val_opt_steps = self.args.optimizer_steps
        # unfortunately need self.avg_num_opt_steps before we can make the path
        self._set_pathes()
        self.meta_logger = create_logger(self, file_handler=True)
        self.meta_logger.info("Initializing experiment - may take a while to load validation set")
        self.fixed_weights = generate_fixed_weights(self)

        if self.args.eval_freq != 0:
            val_funcs = load_val_data(num_of_funcs=self.config.num_val_funcs, n_samples=self.args.x_samples,
                                      stddev=self.config.stddev, dim=self.args.x_dim, logger=self.meta_logger,
                                      exper=self)
        else:
            val_funcs = None

        return val_funcs

    def _set_pathes(self):
        if self.args.log_dir == 'default':
            self.args.log_dir = self.config.exper_prefix + \
                                 str.replace(datetime.now(timezone('Europe/Berlin')).strftime(
                                     '%Y-%m-%d %H:%M:%S.%f')[:-7],
                                             ' ', '_') + "_" + create_exper_label(self) + \
                                 "_lr" + "{:.0e}".format(self.args.lr) + "_" + self.args.optimizer
            self.args.log_dir = str.replace(str.replace(self.args.log_dir, ':', '_'), '-', '')

        else:
            # custom log dir
            self.args.log_dir = str.replace(self.args.log_dir, ' ', '_')
            self.args.log_dir = str.replace(str.replace(self.args.log_dir, ':', '_'), '-', '')
        log_dir = os.path.join(self.config.log_root_path, self.args.log_dir)
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)
            fig_path = os.path.join(log_dir, self.config.figure_path)
            os.makedirs(fig_path)
        else:
            # make a back-up copy of the contents
            dst = os.path.join(log_dir, "backup")
            shutil.copytree(log_dir, dst)

        self.output_dir = log_dir

    def save(self, file_name=None):
        if file_name is None:
            file_name = "exp_statistics" + ".dll"

        outfile = os.path.join(self.output_dir, file_name)

        with open(outfile, 'wb') as f:
            dill.dump(self, f)
        print("INFO - Successfully saved experimental details to {}".format(outfile))

    def end(self, model):
        if not self.args.learner == 'manual' and model.name is not None:
            model_path = os.path.join(self.output_dir, model.name + config.save_ext)
            torch.save(model.state_dict(), model_path)
            self.model_path = model_path
            print("INFO - Successfully saved model parameters to {}".format(model_path))

        self.save()
        if not self.args.on_server:
            loss_plot(self, loss_type="loss", save=True, validation=self.run_validation)
            loss_plot(self, loss_type="opt_loss", save=True, validation=self.run_validation, log_scale=False)
            plot_image_training_loss(self, do_save=True)
            if self.args.learner == "act":
                # plot histogram of T distribution (number of optimization steps during training)
                # plot_dist_optimization_steps(experiment, data_set="train", save=True)
                # plot_dist_optimization_steps(experiment, data_set="val", save=True)
                plot_qt_probs(self, data_set="train", save=True)
                # plot_qt_probs(experiment, data_set="val", save=True, plot_prior=True, height=8, width=8)
                # param_error_plot(experiment, save=True)
        else:
            plot_image_training_loss(self, do_save=True)


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
            prior_probs = construct_prior_p_t_T(steps, exper.config.ptT_shape_param,
                                                batch_size=1, cuda=exper.args.cuda)
            fixed_weights = prior_probs.squeeze()

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


class Epoch(object):

    def __init__(self, exper):
        # want to start at 1
        self.epoch_id = 0
        self.start_time = time.time()
        self.loss_last_time_step = 0.0
        self.final_act_loss = 0.
        self.param_loss = 0.
        self.total_loss_steps = 0.
        self.loss_optimizer = 0.
        self.diff_min = 0.
        self.duration = 0.
        self.avg_opt_steps = []
        self.num_of_batches = exper.args.functions_per_epoch // exper.args.batch_size
        self.prior_probs = construct_prior_p_t_T(exper.args.optimizer_steps, exper.config.ptT_shape_param,
                                                 exper.args.batch_size, exper.args.cuda)
        self.backward_ones = torch.ones(exper.args.batch_size)
        self.max_time_steps_taken = 0
        if exper.args.cuda:
            self.backward_ones = self.backward_ones.cuda()

    def add_step_loss(self, avg_loss, last_time_step=False):
        if not isinstance(avg_loss, (np.float, np.float32, np.float64)):
            raise ValueError("avg_loss must be a numpy.float but is type {}".format(type(avg_loss)))
        self.total_loss_steps += avg_loss
        if last_time_step:
            self.loss_last_time_step += avg_loss

    def add_act_loss(self, loss):
        if not isinstance(loss, (np.float, np.float32, np.float64)):
            raise ValueError("loss must be a numpy.float but is type {}".format(type(loss)))
        self.final_act_loss += loss
        self.loss_optimizer += loss

    def start(self):
        self.epoch_id += 1
        self.start_time = time.time()
        self.loss_last_time_step = 0
        self.final_act_loss = 0
        self.param_loss = 0.
        self.total_loss_steps = 0.
        self.loss_optimizer = 0
        self.diff_min = 0.
        self.duration = 0.
        self.avg_opt_steps = []
        # prepare epoch variables

    def end(self, exper):
        self.duration = time.time() - self.start_time

        exper.meta_logger.info("Epoch: {}, elapsed time {:.2f} seconds: avg optimizer loss {:.4f} / "
                               "avg total loss (over time-steps) {:.4f} /"
                               " avg final step loss {:.4f} / final-true_min {:.4f}".format(self.epoch_id ,
                                                                                            self.duration,
                                                                                            self.loss_optimizer,
                                                                                            self.total_loss_steps,
                                                                                            self.loss_last_time_step,
                                                                                            self.diff_min))
        if exper.args.learner == 'act':
            exper.meta_logger.info("Epoch: {}, ACT - average final act_loss {:.4f}".format(self.epoch_id,
                                                                                           self.final_act_loss))
            avg_opt_steps = int(np.mean(np.array(self.avg_opt_steps)))
            exper.meta_logger.debug("Epoch: {}, Average number of optimization steps {}".format(self.epoch_id + 1,
                                                                                                avg_opt_steps))
        if exper.args.learner == 'meta' and exper.args.version[0:2] == "V2":
            avg_opt_steps = int(np.mean(np.array(self.avg_opt_steps)))
            exper.meta_logger.info("Epoch: {}, Average number of optimization steps {}".format(self.epoch_id,
                                                                                               avg_opt_steps))
        if exper.args.learner == 'act_sb':
            np_array = exper.epoch_stats["halting_step"][self.epoch_id]
            num_of_steps = np_array.shape[0]
            num_of_funcs = np.sum(np_array)
            values = np.arange(0, num_of_steps)
            avg_opt_steps = int(np.sum(1./num_of_funcs * values * np_array))
            E_x_2 = np.sum(1. / num_of_funcs * values**2 * np_array)
            stddev = np.sqrt(E_x_2 - avg_opt_steps**2)
            cum_sum = np.cumsum(np_array)
            if cum_sum[np.nonzero(cum_sum)[0][0]] > num_of_funcs/2.:
                median = np.nonzero(cum_sum)[0][0]
            else:
                median = np.argmax(cum_sum[cum_sum < num_of_funcs/2.]) + 1
            print(exper.epoch_stats["step_losses"][self.epoch_id][0:self.max_time_steps_taken+1])
            print(np_array[0:self.max_time_steps_taken+1])
            # median = int(np.median(np_array))
            exper.meta_logger.info("Epoch: {}, Average number of optimization steps {} "
                                   "stddev {:.3f} median {}".format(self.epoch_id, avg_opt_steps, stddev, median))

        exper.epoch_stats["loss"].append(self.total_loss_steps)
        exper.epoch_stats["param_error"].append(self.param_loss)
        if exper.args.learner == 'act':
            exper.epoch_stats["opt_loss"].append(self.final_act_loss)
        elif exper.args.learner == 'meta':
            exper.epoch_stats["opt_loss"].append(self.loss_optimizer)

    def set_max_time_steps_taken(self, steps):
        self.max_time_steps_taken = steps

    def get_max_time_steps_taken(self):
        return self.max_time_steps_taken
