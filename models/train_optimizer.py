import argparse
import time
import logging

import torch
import torch.optim as optim
from quadratic import Quadratic, Quadratic2D
from rnn_optimizer import MetaLearner, HelperQuadratic


"""
    TODO:
    (1) fixed validation set
    (2) final loss, remember you're nearly always left with a loss at the minimum
        except we define loss as |true - estimated x-value min|, may be better idea?
    (3) current version does not work with pytorch 0.1.10-post2, grads problem
    (4) save the trained optimizer and enable re-training of an existing model
    (5) visualize the error of the meta-learner after training
"""
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
parser.add_argument('--max_epoch', type=int, default=30, metavar='N',
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
args = parser.parse_args()
args.cuda = args.use_cuda and torch.cuda.is_available()
args.cuda = False
args.q2D = True

assert args.optimizer_steps % args.truncated_bptt_step == 0


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(args).items():
        print(key + ' : ' + str(value))


def validate_optimizer(meta_learner, plot_func=False):
    print("------------------------------")
    print("INFO - Validating meta-learner")
    if args.q2D:
        q2_func = Quadratic2D(use_cuda=args.cuda)
        print("Start-value parameters ({:.2f},{:.2f}), true min ({:.2f},{:.2f})".format(
            q2_func.parameter[0].data.numpy()[0], q2_func.parameter[1].data.numpy()[0],
            q2_func.true_opt[0].data.numpy()[0], q2_func.true_opt[1].data.numpy()[0]))
    else:
        q2_func = Quadratic(use_cuda=args.cuda)
        print("Start-value parameter {}, true min {}".format(q2_func.parameter.squeeze().data[0],
                                                             q2_func.true_opt.squeeze().data[0]))
    forward_steps = 0
    for i in range(40):
        # Keep states for truncated BPTT
        if i > args.truncated_bptt_step - 1:
            keep_states = True
        else:
            keep_states = False
        if i % args.truncated_bptt_step == 0:
            forward_steps = 1
            meta_learner.reset_lstm(q2_func, keep_states=keep_states)
        else:
            forward_steps += 1

        loss = q2_func.f_at_xt(hist=True)
        loss.backward()
        delta_p = meta_learner.forward(q2_func.parameter.grad)
        q2_func.parameter.grad.data.zero_()
        # gradient descent
        par_new = q2_func.parameter.data - delta_p.data
        q2_func.parameter.data.copy_(par_new)
        # print("New param-value={}".format(par_new[0]))
    loss_diff = torch.abs(loss.squeeze().data - q2_func.min_value.data)
    if plot_func:
        q2_func.plot_func(show=False, do_save=True)
    if args.q2D:
        print("Final parameter values ({:.2f},{:.2f})".format(q2_func.parameter[0].data.numpy()[0],
                                                            q2_func.parameter[1].data.numpy()[0]))
    else:
        print("Final parameter values {:.2f}".format(q2_func.parameter.data.numpy()[0]))

    print("Validation - final error: {:.4}".format(loss_diff[0]))


def main():
    # set manual seed for random number generation
    SEED = 4325
    torch.manual_seed(SEED)

    print_flags()
    if args.q2D:
        q2_func = Quadratic2D(args.cuda)
    else:
        q2_func = Quadratic(args.cuda)
    meta_optimizer = MetaLearner(HelperQuadratic(q2_func), num_layers=args.num_layers, num_hidden=args.hidden_size,
                                 use_cuda=args.cuda)

    if args.cuda:
        meta_optimizer.cuda()

    optimizer = optim.Adam(meta_optimizer.parameters(), lr=1e-3)

    for epoch in range(args.max_epoch):
        start_epoch = time.time()
        final_loss = 0.0
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

                loss = q2_func.f_at_xt()
                loss.backward()

                # feed the RNN with the gradient of the error surface function
                # meta_q2func = meta_optimizer.meta_update(q2_func)
                # loss_meta = meta_q2func.f_at_xt()
                # set grads of loss-func to zero
                loss_meta = meta_optimizer.meta_update_v2(q2_func)
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

            if i % 20 == 0 and i != 0:
                error = torch.abs(loss.data - q2_func.min_value.data)
                print(error.size())
                print("{}-th function: residual error {:.3f}".format(i, error[0]))
            final_loss += torch.abs(loss.data - q2_func.min_value.data)
        # end of an epoch, print summary
        final_loss *= 1./args.functions_per_epoch
        end_epoch = time.time()
        print("Epoch: {}, elapsed time {:.2f} seconds: average final loss {:.4f}".format(epoch, (end_epoch - start_epoch),
                                                                                   final_loss[0]))
        if epoch % args.print_freq == 0 or epoch + 1 == args.max_epoch:
            pass
            # validate_optimizer(meta_optimizer, plot_func=False)


if __name__ == "__main__":
    main()
