import argparse
import time
import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
from utils.quadratic import Quadratic, Quadratic2D
from utils.regression import RegressionFunction

from utils.config import config
from utils.utils import load_val_data, Experiment, prepare, end_run, get_model, print_flags
from utils.utils import create_logger
from utils.probs import TimeStepsDist, ConditionalTimeStepDist
from val_optimizer import validate_optimizer

"""
    TODO:
    (1) cuda seems at least to "run" but takes twice the time of cpu implementation (25 secs <-> 55 secs)
    (2) re-training does not seem to work. at a certain moment the model predicts gibberish. first thought
    this had to do with validation of model but tested without validation and result stays the same.
"""

MAX_VAL_FUNCS = 100
# for standard optimizer which we compare to
STD_OPT_LR = 4e-1
VALID_VERBOSE = False
PLOT_VALIDATION_FUNCS = True
NOISE_SIGMA = 2.5
POLY_DEGREE = 3

OPTIMIZER_DICT = {'sgd': torch.optim.SGD, # Gradient Descent
                  'adadelta': torch.optim.Adadelta, # Adadelta
                  'adagrad': torch.optim.Adagrad, # Adagrad
                  'adam': torch.optim.Adam, # Adam
                  'rmsprop': torch.optim.RMSprop # RMSprop
                  }

# python train_optimizer.py --max_epoch=40 --learner=act --version=V2.13 --eval_freq=5 --lr=1e-4
# --functions_per_epoch=800 --optimizer=adam --comments="" --hidden_size=128

parser = argparse.ArgumentParser(description='PyTorch Meta-learner')

parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='default learning rate for optimizer (default: 1e-3)')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='number of functions per batch (default: 100)')
parser.add_argument('--optimizer_steps', type=int, default=100, metavar='N',
                    help='number of meta optimizer steps (default: 100)')
parser.add_argument('--truncated_bptt_step', type=int, default=20, metavar='N',
                    help='step at which it truncates bptt (default: 20)')
parser.add_argument('--functions_per_epoch', type=int, default=5000, metavar='N',
                    help='updates per epoch (default: 2000)')
parser.add_argument('--x_samples', type=int, default=100, metavar='N',
                    help='number of values to sample from true regression function (default: 100)')
parser.add_argument('--max_epoch', type=int, default=5, metavar='N',
                    help='number of epoch (default: 5)')
parser.add_argument('--hidden_size', type=int, default=20, metavar='N',
                    help='hidden size of the meta optimizer (default: 20)')
parser.add_argument('--num_layers', type=int, default=2, metavar='N',
                    help='number of LSTM layers (default: 2) for all LSTMs')
parser.add_argument('--use_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N',
                    help='frequency print epoch statistics (default: 5)')
parser.add_argument('--model', type=str, default="default",
                    help='model name that will be used for saving the model to file or load if pickle file is present'
                         'in model directory')
parser.add_argument('--log_dir', type=str, default="default",
                    help='log directory under logs')
parser.add_argument('--retrain', action='store_true', default=False,
                    help='retrain an existing model (note should exist in <models> or specific log_dir (.pkl)')
parser.add_argument('--learner', type=str, default="act",
                    help='type of learner to use 1) manual (e.g. Adam) 2) meta 3) act')
parser.add_argument('--version', type=str, default="V1",
                    help='version of the ACT leaner (currently V1 (two separate LSTMS) and V2 (one LSTM)')
parser.add_argument('--optimizer', type=str, default="adam",
                    help='which optimizer to use sgd, adam, adadelta, adagrad, rmsprop')
parser.add_argument('--comments', type=str, default="", help="add comments to describe specific parameter settings")

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
# Cuda is currently not implemented. Takes longer than CPU, even when testing just Variable/Tensors in python
# interpreter
args.cuda = False


def main():
    # set manual seed for random number generation
    if args.retrain:
        SEED = 2345
    else:
        SEED = 4325
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    
    exper = Experiment(args, config)
    # get distribution P(T) over possible number of total timesteps
    if args.learner == "act":
        pt_dist = TimeStepsDist(config.T, config.continue_prob)
        exper.avg_num_opt_steps = pt_dist.mean
    else:
        exper.avg_num_opt_steps = args.optimizer_steps
    # prepare the experiment
    exper.output_dir = prepare(prcs_args=args, exper=exper)
    # get our logger (one to std-out and one to file)
    meta_logger = create_logger(exper, file_handler=True)
    # print the argument flags
    print_flags(args, meta_logger)
    # load the validation functions
    # val_funcs = load_val_data()

    if not args.learner == 'manual':
        # Important switch, use meta optimizer (LSTM) which will be trained
        meta_optimizer = get_model(exper, POLY_DEGREE, retrain=args.retrain, logger=meta_logger)
        optimizer = OPTIMIZER_DICT[args.optimizer](meta_optimizer.parameters(), lr=args.lr)
    else:
        # we're using one of the standard optimizers, initialized per function below
        meta_optimizer = None
        optimizer = None

    for epoch in range(args.max_epoch):
        exper.epoch += 1
        start_epoch = time.time()
        final_loss = 0.0
        final_act_loss = 0.
        param_loss = 0.
        # in each epoch we optimize args.functions_per_epoch functions in total, packaged in batches of args.batch_size
        # and therefore ideally functions_per_epoch should be a multiple of batch_size
        # ALSO NOTE:
        #       for ACT models we sample for each batch the horizon T aka the number of optimization steps
        #       THEREFORE we should make sure that we have lots of batches per epoch e.g. 5000 functions and
        #       batch size of 50
        num_of_batches = args.functions_per_epoch // args.batch_size
        for i in range(num_of_batches):
            reg_funcs = RegressionFunction(n_funcs=args.batch_size, n_samples=args.x_samples,
                                           noise_sigma=NOISE_SIGMA, poly_degree=POLY_DEGREE,
                                           use_cuda=args.cuda)
            # if we're using a standard optimizer
            if args.learner == 'manual':
                meta_optimizer = OPTIMIZER_DICT[args.optimizer]([reg_funcs.params], lr=STD_OPT_LR)

            # counter that we keep in order to enable BPTT
            forward_steps = 0
            if args.learner != 'act':
                optimizer_steps = args.optimizer_steps
            else:
                # sample T - the number of timesteps - from our PMF (note prob to continue is set in config object)
                # add one to choice because we actually want values between [1, config.T]
                optimizer_steps = pt_dist.rvs(n=1)[0] + 1
                prior_dist = ConditionalTimeStepDist(T=optimizer_steps, q_prob=config.continue_prob)
                prior_probs = Variable(torch.from_numpy(prior_dist.pmfunc(np.arange(1, optimizer_steps + 1))).float())

            meta_logger.debug("Number of optimization steps {}".format(optimizer_steps))
            qt_param = Variable(torch.zeros(1, 1))
            t_params = reg_funcs.true_params.data.numpy()[0, :]
            # meta_logger.info("------------------------- new function ---------------------")
            for k in range(optimizer_steps):
                # Keep states for truncated BPTT
                if k > args.truncated_bptt_step - 1:
                    keep_states = True
                else:
                    keep_states = False
                if k % args.truncated_bptt_step == 0 and not args.learner == 'manual':
                    # meta_logger.debug("DEBUG@step %d - Resetting LSTM" % k)
                    forward_steps = 1
                    meta_optimizer.reset_lstm(keep_states=keep_states)
                    reg_funcs.reset_params()
                    loss_sum = 0

                else:
                    forward_steps += 1

                loss = reg_funcs.compute_loss(average=True)
                # compute gradients of optimizee which will need for the meta-learner
                loss.backward(torch.ones(args.batch_size))
                # feed the RNN with the gradient of the error surface function
                if args.learner == 'meta':
                    delta_param = meta_optimizer.meta_update(reg_funcs)
                    par_new = reg_funcs.params - delta_param
                    loss_step = meta_optimizer.step_loss(reg_funcs, par_new)
                    loss_sum = loss_sum + loss_step
                    reg_funcs.set_parameters(par_new)
                    n_params = par_new.data.numpy()[0, :]
                    d_params = delta_param.data.numpy()[0, :]
                elif args.learner == 'act':
                    delta_param, delta_qt = meta_optimizer.meta_update(reg_funcs)
                    par_new = reg_funcs.params - delta_param
                    qt_param = qt_param + delta_qt
                    loss_step = meta_optimizer.step_loss(par_new)
                    meta_optimizer.q_t.append(qt_param)
                    loss_sum = loss_sum + loss_step
                    reg_funcs.params.data.copy_(par_new.data)
                else:
                    # we're just using one of the pre-delivered optimizers, update function parameters
                    meta_optimizer.step()
                    # compute loss after update
                    loss_step = reg_funcs.compute_loss(average=False)
                if epoch >= args.max_epoch - 5 and i == 5:
                    meta_logger.info("True ({:.3}/{:.3}) - Delta ({:.3}/{:.3}) - New ({:.3}/{:.3})".format(t_params[0],
                                        t_params[1], d_params[0], d_params[1],
                                        n_params[0], n_params[1]))
                reg_funcs.params.grad.data.zero_()

                if (forward_steps == args.truncated_bptt_step or k == optimizer_steps - 1) \
                        and (args.learner == 'meta' or (args.learner == 'act' and args.version[0:2] == "V1")):

                    loss_sum.backward()
                    # finally update meta_learner parameters
                    optimizer.step()
                    meta_optimizer.zero_grad()

            # compute the final loss error for this function between last loss calculated and function min-value
            # error = torch.abs(loss_step.data - q2_func.min_value.data)
            error = loss_step.data

            # back-propagate ACT loss that was accumulated during optimization steps
            if args.learner == 'act':
                # version 1 of the ACT learner has two separate LSTM parametrizations, one for L2L and
                # the other for ACT. So we have 2 loss-functions. Here we back-prop for the ACT loss for
                # both versions
                if args.version[0:2] == 'V1' or args.version[0:2] == 'V2':
                    # processing ACT loss
                    act_loss = meta_optimizer.final_loss(prior_probs, run_type='train')
                    act_loss.backward()
                    optimizer.step()
                    final_act_loss += act_loss.data
                    # set grads of meta_optimizer to zero after update parameters
                    meta_optimizer.zero_grad()

            # to follow training while being in the process of developing the learner, show some detailed info
            # temporary if i % 20 == 0 and i != 0:
            #    detailed_train_info(meta_logger, q2_func, args, meta_optimizer, i, optimizer_steps, error[0])

            if args.learner == "act":
                meta_optimizer.reset_final_loss()
            elif exper.args.learner == 'meta':
                meta_optimizer.reset_losses()
            # END OF SPECIFIC FUNCTION OPTIMIZATION

            # As evaluation measure we take the final error of the function we minimize (which is our loss) minus the
            # actual min-value of the function (so how far away are we from the actual minimum).
            # We sum this error measure for all functions (default 100) inside one epoch
            final_loss += error
            param_loss += reg_funcs.param_error(average=False).data
        # end of an epoch, calculate average final loss/error and print summary
        final_loss *= 1./args.functions_per_epoch
        param_loss *= 1./args.functions_per_epoch
        final_act_loss *= 1./args.functions_per_epoch
        end_epoch = time.time()

        meta_logger.info("Epoch: {}, elapsed time {:.2f} seconds: average final loss {:.4f} / param-loss {:.4f}".format(
              epoch+1, (end_epoch - start_epoch), final_loss[0], param_loss[0]))
        if args.learner == 'act':
            meta_logger.info("Epoch: {}, ACT - average final act_loss {:.4f}".format(epoch+1, final_act_loss[0]))

        exper.epoch_stats["loss"].append(final_loss[0])
        exper.epoch_stats["param_error"].append(param_loss[0])
        if args.learner == 'act':
            exper.epoch_stats["act_loss"].append(final_act_loss[0])
        if exper.epoch % args.eval_freq == 0 or epoch + 1 == args.max_epoch:
            # pass
            if args.learner == 'manual':
                # the manual (e.g. SGD, Adam will be validated using full number of optimization steps
                opt_steps = args.optimizer_steps
            else:
                opt_steps = config.max_val_opt_steps

            # validate_optimizer(meta_optimizer, exper, val_set=val_funcs, meta_logger=meta_logger,
            #                                    verbose=VALID_VERBOSE,
            #                                    plot_func=PLOT_VALIDATION_FUNCS,
            #                                    max_steps=opt_steps,
            #                                    num_of_plots=config.num_val_plots)
    if args.learner == 'act':

        exper.epoch_stats['qt_hist'] = meta_optimizer.qt_hist
        exper.epoch_stats['opt_step_hist'] = meta_optimizer.opt_step_hist
        # save the results of the validation statistics
        exper.val_stats['qt_hist'] = meta_optimizer.qt_hist_val
        exper.val_stats['opt_step_hist'] = meta_optimizer.opt_step_hist_val

    # end_run(exper, meta_optimizer)

if __name__ == "__main__":
    main()
