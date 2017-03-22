import argparse
import time
import logging
import os
import dill
import datetime

import torch
import torch.optim as optim
from utils.quadratic import Quadratic, Quadratic2D

from utils.config import config
from utils.utils import load_val_data, Experiment, prepare, end_run, get_model

"""
    TODO:
    (1) cuda seems at least to "run" but takes twice the time of cpu implementation (25 secs <-> 55 secs)
    (2) re-training does not seem to work. at a certain moment the model predicts gibberish. first thought
    this had to do with validation of model but tested without validation and result stays the same.
"""

MAX_VAL_FUNCS = 100
# DEBUG, INFO, WARNING, ERROR, CRITICAL
logging.basicConfig()
rootLogger = logging.getLogger(__name__)
rootLogger.setLevel(logging.ERROR)

parser = argparse.ArgumentParser(description='PyTorch Meta-learner')

parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='batch size (default: 32)')
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
parser.add_argument('--print_freq', type=int, default=5, metavar='N',
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

args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
args.cuda = False
args.q2D = True
args.save_log = True
args.save_diff_funcs = True

assert args.optimizer_steps % args.truncated_bptt_step == 0


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(args).items():
        print(key + ' : ' + str(value))


def validate_optimizer_v1(meta_learner, exper, with_shows=True, plot_func=False):
    print("------------------------------")
    print("INFO - Validating meta-learner")
    if args.q2D:
        q_func = Quadratic2D(use_cuda=args.cuda)
        if with_shows:
            print("Start-value parameters ({:.2f},{:.2f})".format(
                q_func.parameter[0].data.numpy()[0], q_func.parameter[1].data.numpy()[0]))
    else:
        q_func = Quadratic(use_cuda=args.cuda)
        if with_shows:
            print("Start-value parameter {}, true min {}".format(q_func.parameter.squeeze().data[0],
                                                      q_func.true_opt.squeeze().data[0]))

    fig_name_prefix = os.path.join(exper.output_dir, os.path.join(config.figure_path, "f" + str(exper.epoch)))
    total_loss = 0
    forward_steps = 0
    f = 0
    param_loss = 0
    for i in range(6):
        # Keep states for truncated BPTT
        if i > args.truncated_bptt_step - 1:
            keep_states = True
        else:
            keep_states = False
        if i % args.truncated_bptt_step == 0:
            forward_steps = 1
            meta_learner.reset_lstm(q_func, keep_states=keep_states)
        else:
            forward_steps += 1

        loss = q_func.f_at_xt(hist=True)
        if i % 2 == 0:
            print("Current loss {:.4f}".format(loss.squeeze().data[0]))
        loss.backward()
        delta_p = meta_learner.forward(q_func.parameter.grad)
        q_func.parameter.grad.data.zero_()
        # gradient descent
        par_new = q_func.parameter.data - delta_p.data
        q_func.parameter.data.copy_(par_new)
        # print("New param-value={}".format(par_new[0]))

    if plot_func:
        q_func.plot_func(fig_name=fig_name_prefix, show=False, do_save=True)
    loss_diff = torch.abs(loss.squeeze().data - q_func.min_value.data)
    param_loss += q_func.compute_error()
    if args.q2D:
        if with_shows:
            print("*** {}-th validation function ***".format(f + 1))
            print("True parameter values ({:.2f},{:.2f})".format(q_func.true_opt[0].data.numpy()[0],
                                                                 q_func.true_opt[1].data.numpy()[0]))
            print("Final parameter values ({:.2f},{:.2f})".format(q_func.parameter[0].data.numpy()[0],
                                                                  q_func.parameter[1].data.numpy()[0]))
    else:
        if with_shows:
            print("Final parameter values {:.2f}".format(q_func.parameter.data.numpy()[0]))
    if with_shows:
        print("Validation - final error: {:.4}".format(loss_diff[0]))
    total_loss += loss_diff

    total_loss *= 1. / (f + 1)
    total_param_loss = 1. / (f + 1) * param_loss
    exper.val_stats["loss"].append(total_loss[0])
    exper.val_stats["param_error"].append(total_param_loss)
    print("Final validation average error: {:.4}".format(total_loss[0]))


def validate_optimizer(meta_learner, funcs, exper, plot_func=False, with_shows=False):
    print("------------------------------------------------------")
    print("INFO - Validating meta-learner")

    forward_steps = 0
    total_loss = 0
    total_param_loss = 0
    for f, q_func in enumerate(funcs):
        param_loss = 0.
        if args.q2D:
            q_func.use_cuda = args.cuda
            if with_shows:
                print("Start-value parameters ({:.2f},{:.2f})".format(
                    q_func.parameter[0].data.numpy()[0], q_func.parameter[1].data.numpy()[0]))
        else:
            q_func.use_cuda = args.cuda
            if with_shows:
                print("Start-value parameter {}, true min {}".format(q_func.parameter.squeeze().data[0],
                                                                 q_func.true_opt.squeeze().data[0]))
        for i in range(20):
            # Keep states for truncated BPTT
            if i > args.truncated_bptt_step - 1:
                keep_states = True
            else:
                keep_states = False
            if i % args.truncated_bptt_step == 0:
                forward_steps = 1
                meta_learner.reset_lstm(q_func, keep_states=keep_states)
            else:
                forward_steps += 1

            loss = q_func.f_at_xt(hist=True)
            loss.backward()
            if with_shows:
                if i % 10 == 0 or i+1 == args.optimizer_steps:
                    curr_loss = torch.abs(loss.squeeze().data - q_func.min_value.data)
                    print("INFO-val - {}-step loss {:.4f}".format(i, curr_loss[0]))
            delta_p = meta_learner.forward(q_func.parameter.grad)
            q_func.parameter.grad.data.zero_()
            # gradient descent
            curr_params = q_func.parameter.data
            if args.cuda:
                curr_params = curr_params.cuda()

            par_new = curr_params - delta_p.data
            q_func.parameter.data.copy_(par_new)
            # print("New param-value={}".format(par_new[0]))
        loss_diff = torch.abs(loss.squeeze().data - q_func.min_value.data)
        param_loss += q_func.compute_error()

        if plot_func and f % 50 == 0:
            q_func.plot_func(show=False, do_save=True)
        if args.q2D:
            if with_shows:
                print("*** {}-th validation function ***".format(f+1))
                print("True parameter values ({:.2f},{:.2f})".format(q_func.true_opt[0].data.numpy()[0],
                                                                     q_func.true_opt[1].data.numpy()[0]))
                print("Final parameter values ({:.2f},{:.2f})".format(q_func.parameter[0].data.numpy()[0],
                                                                      q_func.parameter[1].data.numpy()[0]))
        else:
            if with_shows:
                print("Final parameter values {:.2f}".format(q_func.parameter.data.numpy()[0]))
        if with_shows:
            print("Validation - final error: {:.4}".format(loss_diff[0]))
        total_loss += loss_diff
    total_loss *= 1./(f+1)
    total_param_loss = 1./(f+1) * param_loss
    exper.val_stats["loss"].append(total_loss[0])
    exper.val_stats["param_error"].append(total_param_loss)
    print("Final validation average error: {:.4}".format(total_loss[0]))


def main():
    # set manual seed for random number generation
    SEED = 4325
    torch.manual_seed(SEED)
    print_flags()
    exper = Experiment(args)
    val_funcs = load_val_data()
    exper.output_dir = prepare(prcs_args=args)
    # list that stores functions with high residual error during training
    diff_func_list = []

    meta_optimizer = get_model(exper, retrain=args.retrain)

    optimizer = optim.Adam(meta_optimizer.parameters(), lr=1e-3)

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

            forward_steps = 0
            for k in range(args.optimizer_steps):
                # Keep states for truncated BPTT
                if k > args.truncated_bptt_step - 1:
                    keep_states = True
                else:
                    keep_states = False
                if k % args.truncated_bptt_step == 0:
                    rootLogger.debug("INFO@step %d - Resetting LSTM" % k)
                    forward_steps = 1
                    meta_optimizer.reset_lstm(q2_func, keep_states=keep_states)
                    loss_sum = 0

                else:
                    forward_steps += 1

                # loss = 0.5 * (q2_func.f_at_xt() - q2_func.min_value)**2
                loss = q2_func.f_at_xt()
                loss.backward()
                # feed the RNN with the gradient of the error surface function
                # new_params = meta_optimizer.meta_update(q2_func)
                # loss_meta = torch.sum(meta_optimizer.opt_wrapper.optimizee.func(new_params))
                # set grads of loss-func to zero
                loss_meta = meta_optimizer.meta_update(q2_func, normal=False)
                q2_func.parameter.grad.data.zero_()
                loss_sum = loss_sum + loss_meta
                if forward_steps == args.truncated_bptt_step or k == args.optimizer_steps - 1:
                    rootLogger.debug("INFO@step %d(%d) - Backprop for LSTM" % (k, forward_steps))
                    # set grads of meta_optimizer to zero after update parameters
                    meta_optimizer.zero_grad()
                    # average the loss over the optimization steps
                    loss_sum = 1./args.truncated_bptt_step * loss_sum
                    loss_sum.backward()
                    # print("INFO - BPTT LSTM: grads {:.3f}".format(meta_optimizer.sum_grads))
                    # finally update meta_learner parameters
                    optimizer.step()
            error = torch.abs(loss.data - q2_func.min_value.data)
            if error[0] > 1000.:
                print("ERROR - {:.3f}".format(error[0]))
                print("ERROR - function {}".format(q2_func.poly_desc))
                diff_func_list.append(q2_func)
            if i % 20 == 0 and i != 0:

                print("INFO-track -----------------------------------------------------")
                print("{}-th function: residual error {:.3f}".format(i, error[0]))
                print(q2_func.poly_desc)
                if args.q2D:
                    print("Initial parameter values ({:.2f},{:.2f})".format(q2_func.initial_parms[0].data.numpy()[0],
                                                                            q2_func.initial_parms[1].data.numpy()[0]))
                    print("True parameter values ({:.2f},{:.2f})".format(q2_func.true_opt[0].data.numpy()[0],
                                                                         q2_func.true_opt[1].data.numpy()[0]))
                    print("Final parameter values ({:.2f},{:.2f})".format(q2_func.parameter[0].data.numpy()[0],
                                                                          q2_func.parameter[1].data.numpy()[0]))
                else:
                    print("Final parameter values {:.2f}".format(q2_func.parameter.data.numpy()[0]))

            if error[0] > 2. and len(diff_func_list) < MAX_VAL_FUNCS:
                pass
                # diff_func_list.append(q2_func)
            # As evaluation measure we take the final error of the function we minimize (which is our loss) minus the
            # actual min-value of the function (so how far away are we from the actual minimum).
            # We sum this error measure for all functions (default 100) inside one epoch
            final_loss += torch.abs(loss.data - q2_func.min_value.data)
            param_loss += q2_func.compute_error()
        # end of an epoch, calculate average final loss/error and print summary
        final_loss *= 1./args.functions_per_epoch
        param_loss *= 1./args.functions_per_epoch
        end_epoch = time.time()
        print("Epoch: {}, elapsed time {:.2f} seconds: average final loss {:.4f} / param-loss {:.4f}".format(
              epoch+1, (end_epoch - start_epoch), final_loss[0], param_loss))
        exper.epoch_stats["loss"].append(final_loss[0])
        exper.epoch_stats["param_error"].append(param_loss)
        if epoch % args.print_freq == 0 or epoch + 1 == args.max_epoch:
            # pass
            validate_optimizer_v1(meta_optimizer, exper, with_shows=True, plot_func=True)
            # validate_optimizer(meta_optimizer, val_funcs, exper, plot_func=False, with_shows=False)

    end_run(exper, meta_optimizer, diff_func_list)

if __name__ == "__main__":
    main()
