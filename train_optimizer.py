import argparse
import numpy as np

import torch

from utils.config import config
from utils.utils import Experiment, get_model, print_flags
from utils.utils import create_logger
from utils.utils import get_batch_functions
from utils.utils import OPTIMIZER_DICT, Epoch
from train_batch_meta_act import execute_batch

from val_optimizer import validate_optimizer

"""
    The following models can be run for the following problems/tasks:
    ----------------------------------------------------------------------------------------------------------------
    learner             version                 problem                                     comments

    meta                V1                      quadratic, rosenbrock, regression(_T)       baseline model L2L
    meta                V2                      same as V1                                  baseline + stochastic
                                                                                            learning
    meta                V3.1                    same as V1                                  baseline + geometric weights
    meta                V3.2                    same as V1                                  baseline + uniform weights
    meta                V4                      same as V1                                  baseline + learn loss-weights
    meta                V5                      same as V1                                  baseline + ValueFunction
    meta                V6                      only regression(_T)                         baseline with improvement
                                                                                            instead of func-loss
    act                 V1                      regression(_T)                              act learner with 2
                                                                                            separate LSTMS
    act                 V2                      regression(_T)                              act learner with 1 LSTM
"""

# for standard optimizer which we compare to

VALID_VERBOSE = False
TRAIN_VERBOSE = False
PLOT_VALIDATION_FUNCS = False
ANNEAL_LR = False

# python train_optimizer.py --max_epoch=10 --learner=act --version=V2 --lr=4e-6 --batch_size=125 --hidden_size=40
# --functions_per_epoch=10000 --use_cuda --eval_freq=10 --optimizer_steps=100 --problem="regression"
# --fixed_horizon --optimizer_steps=100

parser = argparse.ArgumentParser(description='PyTorch Meta-learner')

parser.add_argument('--x_dim', type=int, default=10, metavar='N',
                    help='dimensionality of the regression variable x (default: 10)')
parser.add_argument('--lr', type=float, default=1e-5, metavar='N',
                    help='default learning rate for optimizer (default: 1e-5)')
parser.add_argument('--batch_size', type=int, default=125, metavar='N',
                    help='number of functions per batch (default: 125)')
parser.add_argument('--optimizer_steps', type=int, default=100, metavar='N',
                    help='number of meta optimizer steps (default: 100)')
parser.add_argument('--truncated_bptt_step', type=int, default=20, metavar='N',
                    help='step at which it truncates bptt (default: 20)')
parser.add_argument('--functions_per_epoch', type=int, default=10000, metavar='N',
                    help='updates per epoch (default: 10000)')
parser.add_argument('--x_samples', type=int, default=10, metavar='N',
                    help='number of values to sample from true regression function (default: 10)')
parser.add_argument('--max_epoch', type=int, default=5, metavar='N',
                    help='number of epoch (default: 5)')
parser.add_argument('--hidden_size', type=int, default=20, metavar='N',
                    help='hidden size of the meta optimizer (default: 20)')
parser.add_argument('--num_layers', type=int, default=2, metavar='N',
                    help='number of LSTM layers (default: 2) for all LSTMs')
parser.add_argument('--use_cuda', action='store_true', default=False,
                    help='shifts tensors to GPU')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N',
                    help='frequency print epoch statistics (default: 5)')
parser.add_argument('--model', type=str, default="default",
                    help='model name that will be used for saving the model to file or load if pickle file is present'
                         'in model directory')
parser.add_argument('--log_dir', type=str, default="default",
                    help='log directory under logs')
parser.add_argument('--checkpoint_dir', type=str, default=None,
                    help='checkpoint directory under default process directory')
parser.add_argument('--checkpoint_eval', type=int, default=20, metavar='N',
                    help='interval between model checkpoint savings (default: 20)')
parser.add_argument('--retrain', action='store_true', default=False,
                    help='retrain an existing model (note should exist in <models> or specific log_dir (.pkl)')
parser.add_argument('--learner', type=str, default="act",
                    help='type of learner to use 1) manual (e.g. Adam) 2) meta 3) act')
parser.add_argument('--version', type=str, default="V1",
                    help='version of the ACT leaner (currently V1 (two separate LSTMS) and V2 (one LSTM)')
parser.add_argument('--optimizer', type=str, default="adam",
                    help='which optimizer to use sgd, adam, adadelta, adagrad, rmsprop')
parser.add_argument('--comments', type=str, default="", help="add comments to describe specific parameter settings")
parser.add_argument('--problem', type=str, default="regression", help="kind of optimization problem (default quadratic")
parser.add_argument('--fixed_horizon', action='store_true', default=False,
                    help='applicable for ACT-model: model will use fixed training horizon (default optimizer_steps)')
parser.add_argument('--on_server', action='store_true', default=False, help="enable if program runs on das4 server")


parser.add_argument("--output_bias")
args = parser.parse_args()
args.output_bias = False
args.cuda = args.use_cuda and torch.cuda.is_available()


def main():
    # set manual seed for random number generation
    if args.retrain:
        SEED = 2345
    else:
        SEED = 4325
    torch.manual_seed(SEED)
    if args.cuda:
        torch.cuda.manual_seed(SEED)

    np.random.seed(SEED)

    exper = Experiment(args, config)
    # get our logger (one to std-out and one to file)
    meta_logger = create_logger(exper, file_handler=True)
    epoch_obj = Epoch(exper, meta_logger)
    # print the argument flags
    print_flags(exper, meta_logger)
    # Initialize EVERYTHING? i.e. if necessary load the validation functions
    val_funcs = exper.start(epoch_obj, meta_logger)

    lr = exper.args.lr
    if not exper.args.learner == 'manual':
        meta_optimizer = get_model(exper, exper.args.x_dim, retrain=exper.args.retrain, logger=meta_logger)
        optimizer = OPTIMIZER_DICT[exper.args.optimizer](meta_optimizer.parameters(), lr=lr)
    else:
        # we're using one of the standard optimizers, initialized per function below
        meta_optimizer = None
        optimizer = None

    for epoch in range(exper.args.max_epoch):
        exper.epoch += 1
        epoch_obj.start()
        exper.epoch_stats["step_losses"][exper.epoch] = np.zeros(exper.max_time_steps + 1)
        exper.epoch_stats["opt_step_hist"][exper.epoch] = np.zeros(exper.max_time_steps + 1)

        for i in range(epoch_obj.num_of_batches):
            reg_funcs = get_batch_functions(exper, exper.config.stddev)
            execute_batch(exper, reg_funcs, meta_optimizer, optimizer, epoch_obj)

        # END OF EPOCH, calculate average final loss/error and print summary
        # we computed the average loss per function in the batch! and added those losses to the final loss
        # therefore we only need to divide through the number of batches to end up with the average loss per function
        epoch_obj.loss_last_time_step *= 1./float(epoch_obj.num_of_batches)
        epoch_obj.param_loss *= 1./float(epoch_obj.num_of_batches)
        epoch_obj.final_act_loss *= 1./float(epoch_obj.num_of_batches)
        epoch_obj.total_loss_steps *= 1./float(epoch_obj.num_of_batches)
        epoch_obj.loss_optimizer *= 1./float(epoch_obj.num_of_batches)
        step_loss_factors = np.where(exper.epoch_stats["opt_step_hist"][exper.epoch] > 0,
                                     1./exper.epoch_stats["opt_step_hist"][exper.epoch], 0)
        exper.epoch_stats["step_losses"][exper.epoch] *= step_loss_factors
        epoch_obj.end(exper)

        # if applicable, VALIDATE model performance
        if exper.run_validation and (exper.epoch % exper.args.eval_freq == 0 or epoch + 1 == exper.args.max_epoch):

            if exper.args.learner == 'manual':
                # the manual (e.g. SGD, Adam will be validated using full number of optimization steps
                opt_steps = exper.args.optimizer_steps
            else:
                opt_steps = exper.config.max_val_opt_steps

            validate_optimizer(meta_optimizer, exper, val_set=val_funcs, meta_logger=meta_logger,
                               verbose=VALID_VERBOSE,
                               plot_func=PLOT_VALIDATION_FUNCS,
                               max_steps=opt_steps,
                               num_of_plots=exper.config.num_val_plots,
                               save_qt_prob_funcs=True if epoch + 1 == exper.args.max_epoch else False,
                               save_model=True)
        # per epoch collect the statistics w.r.t q(t|T) distribution for training and validation
        if exper.args.learner == 'act':

            exper.epoch_stats['qt_hist'][exper.epoch] = meta_optimizer.qt_hist
            exper.epoch_stats['opt_step_hist'][exper.epoch] = meta_optimizer.opt_step_hist

            meta_optimizer.init_qt_statistics(exper.config)
        if hasattr(meta_optimizer, "epochs_trained"):
            meta_optimizer.epochs_trained += 1

    exper.end(meta_optimizer)

if __name__ == "__main__":
    main()
