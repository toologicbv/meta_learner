import argparse
import sys

import numpy as np
import torch

if "/home/jogi/.local/lib/python2.7/site-packages" in sys.path:
    sys.path.remove("/home/jogi/.local/lib/python2.7/site-packages")

from utils.experiment import Experiment
from utils.epoch import Epoch
from utils.config import config
from utils.common import get_model, print_flags
from utils.common import get_batch_functions
from utils.common import OPTIMIZER_DICT
from train_batch_meta_act import execute_batch
from utils.batch_handler import ACTBatchHandler



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
    act_sb              V1                      regression(_T)                              act with stick-breaking approach
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
parser.add_argument('--kl_annealing', action='store_true', default=False, help="using KL cost annealing during training")


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
    # Initialize EVERYTHING? i.e. if necessary load the validation functions
    val_funcs = exper.start()
    # print the argument flags
    print_flags(exper)

    if not exper.args.learner == 'manual':
        meta_optimizer = get_model(exper, exper.args.x_dim, retrain=exper.args.retrain, logger=exper.meta_logger)
        optimizer = OPTIMIZER_DICT[exper.args.optimizer](meta_optimizer.parameters(), lr=exper.args.lr)
    else:
        # we're using one of the standard optimizers, initialized per function below
        meta_optimizer = None
        optimizer = None

    for epoch in range(exper.args.max_epoch):
        exper.epoch += 1
        ACTBatchHandler.id = 0
        exper.init_epoch_stats()
        epoch_obj = Epoch(exper)
        epoch_obj.start()
        kl_weight = float(exper.annealing_schedule[epoch])
        exper.meta_logger.info("Epoch: {} - using kl-weight {:.4f}".format(exper.epoch, kl_weight))
        for i in range(epoch_obj.num_of_batches):
            if exper.args.learner in ['meta', 'act']:
                reg_funcs = get_batch_functions(exper)
                execute_batch(exper, reg_funcs, meta_optimizer, optimizer, epoch_obj)
            elif exper.args.learner in ['act_sb']:
                batch = ACTBatchHandler(exper, is_train=True)
                ACTBatchHandler.id += 1
                batch(exper, epoch_obj, meta_optimizer)
                act_loss = batch.backward(epoch_obj, meta_optimizer, optimizer, kl_weight=kl_weight)
                epoch_obj.add_act_loss(act_loss)
            else:
                raise ValueError("args.learner {} not supported by this implementation".format(exper.args.learner))

        # END OF EPOCH, calculate average final loss/error and print summary
        # we computed the average loss per function in the batch! and added those losses to the final loss
        # therefore we only need to divide through the number of batches to end up with the average loss per function

        exper.set_kl_term(epoch_obj.kl_term, kl_weight)
        exper.scale_step_losses()
        epoch_obj.end(exper)
        # if applicable, VALIDATE model performance
        if exper.run_validation and (exper.epoch % exper.args.eval_freq == 0 or epoch + 1 == exper.args.max_epoch):
            exper.eval(epoch_obj, meta_optimizer, val_funcs, save_model=True, save_run=None) # "{}".format(exper.epoch)
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
