import argparse
import time
import logging
import os

import torch
import torch.optim as optim
from utils.quadratic import Quadratic, Quadratic2D

from utils.config import config
from utils.utils import load_val_data, Experiment, prepare, end_run, get_model, print_flags
from utils.utils import create_logger

"""
    TODO:
    (1) cuda seems at least to "run" but takes twice the time of cpu implementation (25 secs <-> 55 secs)
    (2) re-training does not seem to work. at a certain moment the model predicts gibberish. first thought
    this had to do with validation of model but tested without validation and result stays the same.
"""

MAX_VAL_FUNCS = 100
STD_OPT_LR = 4e-1
META_LR = 1e-3
VALID_VERBOSE = False
VALID_PLOT = False

# DEBUG, INFO, WARNING, ERROR, CRITICAL
# logging.basicConfig()
# rootLogger = logging.getLogger(__name__)
# rootLogger.setLevel(logging.ERROR)

parser = argparse.ArgumentParser(description='PyTorch Meta-learner')

parser.add_argument('--optimizer_steps', type=int, default=80, metavar='N',
                    help='number of meta optimizer steps (default: 100)')
parser.add_argument('--truncated_bptt_step', type=int, default=20, metavar='N',
                    help='step at which it truncates bptt (default: 20)')
parser.add_argument('--functions_per_epoch', type=int, default=100, metavar='N',
                    help='updates per epoch (default: 100)')
parser.add_argument('--max_epoch', type=int, default=5, metavar='N',
                    help='number of epoch (default: 10000)')
parser.add_argument('--hidden_size', type=int, default=20, metavar='N',
                    help='hidden size of the meta optimizer (default: 20)')
parser.add_argument('--num_layers', type=int, default=2, metavar='N',
                    help='number of LSTM layers (default: 2)')
parser.add_argument('--use_cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--eval_freq', type=int, default=5, metavar='N',
                    help='frequency print epoch statistics (default: 10)')
parser.add_argument('--q2D', action='store_true', default=False,
                    help='using 2D quadratic functions')
parser.add_argument('--save_log', action='store_true', default=False,
                    help='store experimental run details')
parser.add_argument('--save_diff_funcs', action='store_true', default=False,
                    help='whether or not to save the detected difficult to optimize functions')
parser.add_argument('--model', type=str, default="meta_learner_v1",
                    help='model name that will be used for saving the model to file or load if pickle file is present')
parser.add_argument('--log_dir', type=str, default="default",
                    help='log directory under logs')
parser.add_argument('--retrain', action='store_true', default=False,
                    help='retrain an existing model (note should exist in <models> or specific log_dir (.pkl)')
parser.add_argument('--loss_type', type=str, default="MSE",
                    help='Loss of optimizer: 1) MSE 2) EVAL')
parser.add_argument('--learner', type=str, default="act",
                    help='type of learner to use 1) manual (e.g. Adam) 2) meta 3) act')

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
args.cuda = False
args.q2D = True
args.save_log = True
args.save_diff_funcs = True


def validate_optimizer(meta_learner, exper, meta_logger, val_set=None, steps=6, verbose=True, plot_func=False,
                       num_of_plots=3):
    global STD_OPT_LR
    # we will probably call this procedure later in another context (to evaluate meta-learners)
    # so make sure the globals exist.
    if 'STD_OPT_LR' not in globals():
        meta_logger.debug("create global")
        STD_OPT_LR = 4e-1

    meta_logger.info("-----------------------------------------------------------")
    if val_set is None:
        # if no validation set is provided just use one random generated q-function to run the validation
        val_set = [Quadratic2D(use_cuda=args.cuda)]
        plot_idx = [0]
    else:
        plot_idx = [(i + 1) * (len(val_set) // num_of_plots) - 1 for i in range(num_of_plots)]
    num_of_funcs = len(val_set)

    meta_logger.info("INFO - Epoch {}: Validating meta-learner with {} functions".format(exper.epoch, num_of_funcs))
    total_loss = 0
    total_param_loss = 0
    for f, q_func in enumerate(val_set):
        if verbose and f in plot_idx:
            meta_logger.info("******* {}-th validation function *******".format(f + 1))
        if args.q2D:
            q_func.use_cuda = args.cuda
            q_func.reset()
            if verbose and f in plot_idx:
                meta_logger.info("\tStart-value parameters ({:.2f},{:.2f})".format(
                    q_func.parameter[0].data.numpy()[0], q_func.parameter[1].data.numpy()[0]))
        else:
            q_func.use_cuda = args.cuda
            q_func.reset()
            if verbose and f in plot_idx:
                meta_logger.info("\tStart-value parameter {}, true min {}".format(q_func.parameter.squeeze().data[0],
                                                          q_func.true_opt.squeeze().data[0]))

        if exper.args.learner == 'manual':
            state_dict = meta_learner.state_dict()
            meta_learner = optim.Adam([q_func.parameter], lr=STD_OPT_LR)
            meta_learner.load_state_dict(state_dict)

        for i in range(steps):
            # Keep states for truncated BPTT
            if i > args.truncated_bptt_step - 1:
                keep_states = True
            else:
                keep_states = False
            if i % args.truncated_bptt_step == 0 and not exper.args.learner == 'manual':
                meta_learner.reset_lstm(q_func, keep_states=keep_states)

            loss = q_func.f_at_xt(hist=True)
            if verbose and not exper.args.learner == 'manual' and i % 2 == 0 and f in plot_idx:
                meta_logger.info("\tCurrent loss {:.4f}".format(loss.squeeze().data[0]))
            loss.backward()
            if not exper.args.learner == 'manual':
                delta_p = meta_learner.forward(q_func.parameter.grad)
                if exper.args.learner == 'meta':
                    # gradient descent
                    par_new = q_func.parameter.data - delta_p.data
                elif exper.args.learner == 'act':
                    # in this case forward returns a tuple (parm_delta, qt)
                    par_new = q_func.parameter.data - delta_p[0].data
                q_func.parameter.data.copy_(par_new)
            else:
                meta_learner.step()

            q_func.parameter.grad.data.zero_()

        if plot_func and f in plot_idx:
            fig_name_prefix = os.path.join(exper.output_dir, os.path.join(config.figure_path,
                                                                          str(exper.epoch)+"_f"+str(f+1)))
            q_func.plot_func(fig_name=fig_name_prefix, show=False, do_save=True)
        # compute the losses (1) final loss function (2) final loss parameters (MSE)
        loss_diff = torch.abs(loss.squeeze().data - q_func.min_value.data)
        total_param_loss += q_func.compute_error()
        if args.q2D:
            if verbose and f in plot_idx:
                meta_logger.info("\tTrue parameter values ({:.2f},{:.2f})".format(q_func.true_opt[0].data.numpy()[0],
                                                                     q_func.true_opt[1].data.numpy()[0]))
                meta_logger.info("\tFinal parameter values ({:.2f},{:.2f})".format(q_func.parameter[0].data.numpy()[0],
                                                                      q_func.parameter[1].data.numpy()[0]))
        else:
            if verbose and f in plot_idx:
                meta_logger.info("\tFinal parameter values {:.2f}".format(q_func.parameter.data.numpy()[0]))
        if verbose and f in plot_idx:
            meta_logger.info("\tValidation - final error: {:.4}".format(loss_diff[0]))
        total_loss += loss_diff

    total_loss *= 1. / (f + 1)
    total_param_loss *= 1. / (f + 1)
    exper.val_stats["loss"].append(total_loss[0])
    exper.val_stats["param_error"].append(total_param_loss)
    meta_logger.info("INFO - Epoch {}: Final validation average loss / param-loss: {:.4}/{:.4}".format(exper.epoch,
                                                                                                       total_loss[0],
                                                                                                       total_param_loss))


def main():
    # set manual seed for random number generation
    SEED = 4325
    torch.manual_seed(SEED)
    print_flags(args)
    exper = Experiment(args)
    exper.output_dir = prepare(prcs_args=args)
    meta_logger = create_logger(exper)
    val_funcs = load_val_data()

    # list that stores functions with high residual error during training
    diff_func_list = []
    if not args.learner == 'manual':
        # Important switch, use meta optimizer (LSTM) which will be trained
        meta_optimizer = get_model(exper, retrain=args.retrain)
        optimizer = optim.Adam(meta_optimizer.parameters(), lr=META_LR)
    else:
        # we're using one of the standard optimizers, initialized per function below
        meta_optimizer = None
        optimizer = None

    for epoch in range(args.max_epoch):
        exper.epoch += 1
        start_epoch = time.time()
        final_loss = 0.0
        param_loss = 0.
        # loop over functions we are going to optimize
        for i in range(args.functions_per_epoch):

            if args.q2D:
                q2_func = Quadratic2D(use_cuda=args.cuda)
            else:
                q2_func = Quadratic(use_cuda=args.cuda)
            # if we're using a standard optimizer
            if args.learner == 'manual':
                meta_optimizer = optim.Adam([q2_func.parameter], lr=STD_OPT_LR)

            # counter that we keep in order to enable BPTT
            forward_steps = 0
            for k in range(args.optimizer_steps):
                # Keep states for truncated BPTT
                if k > args.truncated_bptt_step - 1:
                    keep_states = True
                else:
                    keep_states = False
                if k % args.truncated_bptt_step == 0 and not args.learner == 'manual':
                    meta_logger.debug("DEBUG@step %d - Resetting LSTM" % k)
                    forward_steps = 1
                    meta_optimizer.reset_lstm(q2_func, keep_states=keep_states)
                    loss_sum = 0

                else:
                    forward_steps += 1

                loss = q2_func.f_at_xt()
                loss.backward()
                # feed the RNN with the gradient of the error surface function
                # new_params = meta_optimizer.meta_update(q2_func)
                # loss_meta = torch.sum(meta_optimizer.opt_wrapper.optimizee.func(new_params))
                # set grads of loss-func to zero
                if args.learner == 'meta' or args.learner == 'act':
                    loss_meta = meta_optimizer.meta_update(q2_func, loss_type=args.loss_type)
                    loss_sum = loss_sum + loss_meta

                else:
                    # we're just using one of the pre-delivered optimizers, update function parameters
                    meta_optimizer.step()

                q2_func.parameter.grad.data.zero_()

                if (forward_steps == args.truncated_bptt_step or k == args.optimizer_steps - 1) \
                        and not args.learner == 'manual':
                    meta_logger.debug("DEBUG@step %d(%d) - Backprop for LSTM" % (k+1, forward_steps))
                    # average the loss over the optimization steps, PEP complains about not using in-place multi
                    # but pytorch autograd can't handle this (according to documentation)
                    loss_sum *= 1./args.truncated_bptt_step
                    loss_sum.backward()
                    # finally update meta_learner parameters
                    optimizer.step()
                    # set grads of meta_optimizer to zero after update parameters
                    meta_optimizer.zero_grad()

            # compute the final loss error for this function between last loss calculated and function min-value
            error = torch.abs(loss.data - q2_func.min_value.data)
            if not exper.args.learner == 'manual' and error[0] > 20. and epoch != 0 and len(diff_func_list) < MAX_VAL_FUNCS:
                # add this function to the list of "difficult" functions. can be used later for analysis
                # and will be saved in log directory as "dill" dump
                diff_func_list.append(q2_func)
            if i % 20 == 0 and i != 0:

                meta_logger.info("INFO-track -----------------------------------------------------")
                meta_logger.info("{}-th function: loss {:.4f}".format(i, error[0]))
                meta_logger.info(q2_func.poly_desc)
                if args.q2D:
                    meta_logger.info("Initial parameter values ({:.2f},{:.2f})".format(
                        q2_func.initial_parms[0].data.numpy()[0],
                        q2_func.initial_parms[1].data.numpy()[0]))
                    meta_logger.info("True parameter values ({:.2f},{:.2f})".format(
                        q2_func.true_opt[0].data.numpy()[0],
                        q2_func.true_opt[1].data.numpy()[0]))
                    meta_logger.info("Final parameter values ({:.2f},{:.2f})".format(
                        q2_func.parameter[0].data.numpy()[0],
                        q2_func.parameter[1].data.numpy()[0]))
                else:
                    meta_logger.info("Final parameter values {:.2f}".format(q2_func.parameter.data.numpy()[0]))
            # END OF SPECIFIC FUNCTION OPTIMIZATION
            if args.learner == 'act':
                meta_logger.debug("**************************** act loss backward **********************************")
                act_loss = meta_optimizer.final_loss()
                act_loss.backward()
                optimizer.step()
                # set grads of meta_optimizer to zero after update parameters
                meta_optimizer.zero_grad()
                meta_optimizer.reset_final_loss()
            # As evaluation measure we take the final error of the function we minimize (which is our loss) minus the
            # actual min-value of the function (so how far away are we from the actual minimum).
            # We sum this error measure for all functions (default 100) inside one epoch
            final_loss += torch.abs(loss.data - q2_func.min_value.data)
            param_loss += q2_func.compute_error()
        # end of an epoch, calculate average final loss/error and print summary
        final_loss *= 1./args.functions_per_epoch
        param_loss *= 1./args.functions_per_epoch
        end_epoch = time.time()
        meta_logger.info("Epoch: {}, elapsed time {:.2f} seconds: average final loss {:.4f} / param-loss {:.4f}".format(
              epoch+1, (end_epoch - start_epoch), final_loss[0], param_loss))
        exper.epoch_stats["loss"].append(final_loss[0])
        exper.epoch_stats["param_error"].append(param_loss)
        if epoch % args.eval_freq == 0 or epoch + 1 == args.max_epoch:
            # pass
            if args.learner == 'manual':
                opt_steps = 6
            else:
                opt_steps = args.optimizer_steps

            validate_optimizer(meta_optimizer, exper, val_set=val_funcs, meta_logger=meta_logger,
                               verbose=VALID_VERBOSE,
                               plot_func=VALID_PLOT,
                               steps=opt_steps, num_of_plots=3)

    end_run(exper, meta_optimizer, diff_func_list)

if __name__ == "__main__":
    main()
