import math
import torch
import os
import dill
import shutil
from config import config
from datetime import datetime
import argparse
import logging

from models.rnn_optimizer import MetaLearner, WrapperOptimizee
from quadratic import Quadratic2D, Quadratic
from plots import loss_plot, param_error_plot


def create_logger(exper):
    # create logger
    logger = logging.getLogger('meta learner')
    logger.setLevel(logging.DEBUG)
    # create file handler which logs even debug messages
    fh = logging.FileHandler(os.path.join(exper.output_dir, config.logger_filename))
    fh.setLevel(logging.INFO)
    # create console handler with a higher log level
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    # create formatter and add it to the handlers
    formatter_fh = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter_ch = logging.Formatter('%(levelname)s - %(message)s')
    fh.setFormatter(formatter_fh)
    ch.setFormatter(formatter_ch)
    # add the handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger


def print_flags(exper_args):
    """
    Prints all entries in argument parser.
    """
    for key, value in vars(exper_args).items():
        print(key + ' : ' + str(value))


def preprocess_gradients(x):
    p = 10
    eps = 1e-6
    indicator = (x.abs() > math.exp(-p)).float()
    x1 = (x.abs() + eps).log() / p * indicator - (1 - indicator)
    x2 = x.sign() * indicator + math.exp(p) * x * (1 - indicator)

    return torch.cat((x1, x2), 1)


def get_experiment(file_to_exp):
    file_to_exp = os.path.join(config.log_root_path, os.path.join(file_to_exp, config.exp_file_name))
    try:
        with open(file_to_exp, 'rb') as f:
            experiment = dill.load(f)
    except:
        raise IOError("Can't open file {}".format(file_to_exp))
    return experiment


def create_def_argparser(**kwargs):

    args = argparse.Namespace()
    args.cuda = kwargs['cuda']
    args.q2D = kwargs['q2D']
    args.model = kwargs['model']
    args.log_dir = kwargs['log_dir']
    args.use_std_opt = False
    args.num_layers = 2
    args.hidden_size = 20
    args.retrain = kwargs['retrain']
    args.loss_type = kwargs['loss_type']

    return args


def get_model(exper, retrain=False):

    if exper.args.q2D:
        q2_func = Quadratic2D(use_cuda=exper.args.cuda)
    else:
        q2_func = Quadratic(use_cuda=exper.args.cuda)
    meta_optimizer = MetaLearner(WrapperOptimizee(q2_func), num_layers=exper.args.num_layers,
                                 num_hidden=exper.args.hidden_size,
                                 use_cuda=exper.args.cuda)
    if retrain:
        loaded = False
        if exper.model_path is not None:
            # try to load model
            if os.path.exists(exper.model_path):
                meta_optimizer.load_state_dict(torch.load(exper.model_path))
                print("INFO - loaded existing model from file {}".format(exper.model_path))
                loaded = True
            else:
                print("Warning - model not found in {}".format(exper.model_path))

        if exper.args.model is not None and not loaded:
            model_file_name = os.path.join(config.model_path, (exper.args.model + config.save_ext))
            if os.path.exists(model_file_name):
                meta_optimizer.load_state_dict(torch.load(model_file_name))
                print("INFO - loaded existing model from file {}".format(model_file_name))
                loaded = True

        if not loaded:
            print("Warning - retrain was enabled but model <{}> does not exist. "
                  "Training a new model with this name.".format(exper.args.model))

    else:
        print("INFO - training a new model {}".format(exper.args.model))
    meta_optimizer.name = exper.args.model

    if exper.args.cuda:
        meta_optimizer.cuda()
    print(meta_optimizer.state_dict().keys())

    return meta_optimizer


def load_val_data(path_specs=None):
    if path_specs is not None:
        load_file = path_specs
    else:
        load_file = os.path.join(config.data_path, config.validation_funcs)
    try:
        with open(load_file, 'rb') as f:
            val_funcs = dill.load(f)
        print("INFO - validation set loaded from {}".format(load_file))
    except:
        raise IOError("Can't open file {}".format(load_file))

    return val_funcs


class Experiment(object):

    def __init__(self, run_args):
        self.args = run_args
        self.epoch_stats = {"loss": [], "param_error": []}
        self.val_stats = {"loss": [], "param_error": []}
        self.epoch = 0
        self.output_dir = None
        self.model_path = None


def save_exper(exper):
    file_name = "exp_statistics" + ".dll"
    outfile = os.path.join(exper.output_dir, file_name)

    with open(outfile, 'wb') as f:
        dill.dump(exper, f)
    print("INFO - Successfully saved experimental details to {}".format(outfile))


def prepare(prcs_args):

    if prcs_args.log_dir == 'default':
        prcs_args.log_dir = config.exper_prefix + str.replace(datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S.%f')[:-7],
                                                              ' ', '_')
        prcs_args.log_dir = str.replace(str.replace(prcs_args.log_dir, ':', '_'), '-', '')
    else:
        prcs_args.log_dir = str.replace(prcs_args.log_dir, ' ', '_')
        prcs_args.log_dir = str.replace(str.replace(prcs_args.log_dir, ':', '_'), '-', '')
    log_dir = os.path.join(config.log_root_path, prcs_args.log_dir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
        fig_path = os.path.join(log_dir, config.figure_path)
        os.makedirs(fig_path)
    else:
        dst = os.path.join(log_dir, "backup")
        shutil.copytree(log_dir, dst)
    return log_dir


def end_run(experiment, model, func_list):
    if not experiment.args.use_std_opt and model.name is not None:
        model_path = os.path.join(experiment.output_dir, model.name + config.save_ext)
        torch.save(model.state_dict(), model_path)
        experiment.model_path = model_path
        print("INFO - Successfully saved model parameters to {}".format(model_path))
    if experiment.args.save_diff_funcs:
        diff_func_file = os.path.join(experiment.output_dir, "diff_funcs.dll")
        with open(diff_func_file, 'wb') as f:
            dill.dump(func_list, f)
        print("INFO - Successfully saved selection of <{}> difficult functions to {}".format(len(diff_func_file),
                                                                                             diff_func_file))
    if experiment.args.save_log:
        save_exper(experiment)
        loss_plot(experiment, save=True)
        param_error_plot(experiment, save=True)
