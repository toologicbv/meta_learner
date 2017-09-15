import argparse
import sys

import numpy as np
import torch
from torch.autograd import Variable

if "/home/jogi/.local/lib/python2.7/site-packages" in sys.path:
    sys.path.remove("/home/jogi/.local/lib/python2.7/site-packages")

from utils.experiment import Experiment
from utils.epoch import Epoch
from utils.config import config
from utils.common import get_model, print_flags, load_curriculum
from utils.common import get_batch_functions
from utils.common import OPTIMIZER_DICT
from train_batch_meta_act import execute_batch
import utils.batch_handler



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
    meta                V7                      regression(_T)                              baseline + incremental learning
    act                 V1                      regression(_T)                              act learner with 2
                                                                                            separate LSTMS
    act                 V2                      regression(_T)                              act learner with 1 LSTM
    act_sb              V1                      regression(_T)                              act with stick-breaking approach
    act_sb              V2                      regression(_T)                              act with SB and KL cost annealing
    meta_act          V1                      regression(_T)                              Graves ACT with ponder-cost
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
parser.add_argument('--checkpoint_dir', type=str, default="checkpoints",
                    help='checkpoint directory under default process directory')
parser.add_argument('--checkpoint_eval', type=int, default=None, metavar='N',
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
parser.add_argument('--problem', type=str, default="regression_T", help="kind of optimization problem "
                                                                        "(default regression_T")
parser.add_argument('--fixed_horizon', action='store_true', default=False,
                    help='applicable for ACT-model: model will use fixed training horizon (default optimizer_steps)')
parser.add_argument('--on_server', action='store_true', default=False, help="enable if program runs on das4 server")
parser.add_argument('--samples_per_batch', type=int, default=1, metavar='N', help='number of samples per batch (default: 1)')
parser.add_argument('--lr_step_decay', type=int, default=0, help="enables learning rate step-decay (after loss_threshold")


parser.add_argument("--output_bias")
args = parser.parse_args()
# Important - we don't use an output bias on the LSTM linear output layer. Experiencing "drifting" behavior if we do
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
        if exper.args.problem == "mlp":
            num_of_inputs = 3
        else:
            num_of_inputs = 1
        meta_optimizer = get_model(exper, num_of_inputs, retrain=exper.args.retrain)
        exper.optimizer = OPTIMIZER_DICT[exper.args.optimizer](meta_optimizer.parameters(), lr=exper.args.lr)
    else:
        # we're using one of the standard optimizers, initialized per function below
        meta_optimizer = None
        exper.optimizer = None

    batch_handler_class = None if exper.batch_handler_class is None else \
        getattr(utils.batch_handler, exper.batch_handler_class)

    if exper.args.learner == "meta" and exper.args.version == "V7":
        curriculum_schedule = load_curriculum("curriculum.dll")
    for epoch in range(exper.args.max_epoch):
        exper.epoch += 1
        batch_handler_class.id = 0
        exper.init_epoch_stats()
        epoch_obj = Epoch()
        epoch_obj.start(exper)
        exper.meta_logger.info("Epoch {}: Num of batches {}".format(exper.epoch, epoch_obj.num_of_batches))
        if exper.args.learner == "meta" and exper.args.version == "V7":
            global_curriculum = curriculum_schedule[exper.epoch - 1]
        for i in range(epoch_obj.num_of_batches):
            if exper.args.learner in ['meta', 'act']:
                # exper.meta_logger.info("Epoch {}: batch {}".format(exper.epoch, i + 1))
                optimizees = get_batch_functions(exper)
                if exper.args.learner == "meta" and exper.args.version == "V7":
                    exper.inc_learning_schedule[exper.epoch - 1] = global_curriculum[i]
                execute_batch(exper, optimizees, meta_optimizer, exper.optimizer, epoch_obj,
                              final_batch=True if i+1 == epoch_obj.num_of_batches else False)

            elif exper.args.learner[0:6] in ['act_sb'] or exper.args.learner == "meta_act":
                loss_sum = Variable(torch.DoubleTensor([0.]))
                kl_sum = 0.
                penalty_sum = 0.
                if exper.args.cuda:
                    loss_sum = loss_sum.cuda()
                for _ in np.arange(exper.args.samples_per_batch):
                    batch = batch_handler_class(exper, is_train=True)
                    batch_handler_class.id += 1
                    # final_batch parameter does nothing else than printing the accuracy for the last batch in the
                    # MLP experiment. Not used for regression_(T)
                    batch(exper, epoch_obj, meta_optimizer, final_batch=True if i+1 == epoch_obj.num_of_batches else False)
                    batch.compute_batch_loss(epoch_obj.weight_regularizer)
                    loss_sum += batch.loss_sum
                    kl_sum += batch.kl_term
                    penalty_sum += batch.penalty_term

                loss_sum = loss_sum * 1./float(exper.args.samples_per_batch)
                act_loss, sum_grads = batch.backward(epoch_obj, meta_optimizer, exper.optimizer, loss_sum=loss_sum)
                epoch_obj.model_grads.append(sum_grads)
                epoch_obj.add_kl_term(kl_sum * 1./float(exper.args.samples_per_batch),
                                      penalty_sum * 1./float(exper.args.samples_per_batch))
                epoch_obj.add_act_loss(act_loss)
            else:
                raise ValueError("args.learner {} not supported by this implementation".format(exper.args.learner))

        # END OF EPOCH, calculate average final loss/error and print summary
        # we computed the average loss per function in the batch! and added those losses to the final loss
        # therefore we only need to divide through the number of batches to end up with the average loss per function

        exper.scale_step_statistics()
        epoch_obj.end(exper)
        # check whether we need to adjust the learning rate
        if exper.args.problem == "mlp" and (exper.args.learner == "meta_act" or exper.args.learner == "act_sb"):
            if exper.args.lr_step_decay != 0 \
                    and (epoch_obj.loss_optimizer <= exper.loss_threshold_lr_decay
                         or exper.lr_decay_last_epoch != 0):
                exper.check_lr_decay(exper, meta_optimizer, epoch_obj.loss_optimizer)
        # execute a checkpoint (save model) if necessary
        if exper.args.checkpoint_eval is not None and exper.epoch % exper.args.checkpoint_eval == 0:
            epoch_obj.execute_checkpoint(exper, meta_optimizer)

        # if applicable, VALIDATE model performance
        if exper.run_validation and (exper.epoch % exper.args.eval_freq == 0 or epoch + 1 == exper.args.max_epoch):
            exper.eval(epoch_obj, meta_optimizer, val_funcs, save_model=True, save_run=None) # "{}".format(exper.epoch)
        # per epoch collect the statistics w.r.t q(t|x, T) distribution for training and validation
        if exper.args.learner == 'act':

            exper.epoch_stats['qt_hist'][exper.epoch] = meta_optimizer.qt_hist
            exper.epoch_stats['opt_step_hist'][exper.epoch] = meta_optimizer.opt_step_hist

            meta_optimizer.init_qt_statistics(exper.config)
        if hasattr(meta_optimizer, "epochs_trained"):
            meta_optimizer.epochs_trained += 1

    exper.end(meta_optimizer)

if __name__ == "__main__":
    main()
