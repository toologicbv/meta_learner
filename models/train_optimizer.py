import argparse
import time

import torch
import torch.optim as optim
from quadratic import Quadratic
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

parser = argparse.ArgumentParser(description='PyTorch Meta-optimizer example')

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
parser.add_argument('--hidden_size', type=int, default=10, metavar='N',
                    help='hidden size of the meta optimizer (default: 10)')
parser.add_argument('--num_layers', type=int, default=2, metavar='N',
                    help='number of LSTM layers (default: 2)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--print_freq', type=int, default=5, metavar='N',
                    help='frequency print epoch statistics (default: 10)')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.cuda = False

assert args.optimizer_steps % args.truncated_bptt_step == 0


def print_flags():
    """
    Prints all entries in FLAGS variable.
    """
    for key, value in vars(args).items():
        print(key + ' : ' + str(value))


def validate_optimizer(meta_learner):
    print("------------------------------")
    print("INFO - Validating meta-learner")
    q2_func = Quadratic(use_cuda=args.cuda)
    initial_loss = q2_func.f_at_xt()
    print("Start-value parameter {}, true min {}".format(q2_func.parameter.squeeze().data[0],
                                                         q2_func.true_opt.squeeze().data[0]))
    forward_steps = 0
    loss_diff = 0.
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

        q2_func.parameter.grad.data.zero_()
        loss = q2_func.f_at_xt(hist=True)
        loss.backward()
        delta_p = meta_learner.forward(q2_func.parameter.grad)
        par_new = q2_func.parameter.data + delta_p.data
        q2_func.parameter.data.copy_(par_new)
        # print("New param-value={}".format(par_new[0]))
        loss_diff = initial_loss.squeeze().data[0] - loss.squeeze().data[0]
    q2_func.plot_func("figures/" + q2_func.poly_desc() + ".png")
    print("Validation error: initial-final={} - {} = {} / final param-value={}".format(initial_loss.squeeze().data[0],
                                                                                loss.squeeze().data[0],
                                                                                loss_diff,
                                                                                q2_func.parameter.squeeze().data[0]))


def main():
    # set manual seed for random number generation
    SEED = 4325
    torch.manual_seed(SEED)

    print_flags()
    q2_func = Quadratic(args.cuda)
    meta_optimizer = MetaLearner(HelperQuadratic(q2_func), num_layers=args.num_layers, num_hidden=args.hidden_size,
                                   use_cuda=args.cuda)

    if args.cuda:
        meta_optimizer.cuda()

    optimizer = optim.Adam(meta_optimizer.parameters(), lr=1e-3)

    for epoch in range(args.max_epoch):
        start_epoch = time.time()
        loss_diff = 0.0
        final_loss = 0.0
        # loop over functions we are going to optimize
        for i in range(args.functions_per_epoch):
            start_func = time.time()
            q2_func = Quadratic(use_cuda=args.cuda)
            initial_loss = q2_func.f_at_xt()
            forward_steps = 0
            for k in range(args.optimizer_steps):
                # Keep states for truncated BPTT
                if k > args.truncated_bptt_step - 1:
                    keep_states = True
                else:
                    keep_states = False
                if k % args.truncated_bptt_step == 0:
                    # print("INFO@step %d - Resetting LSTM" % k)
                    forward_steps = 1
                    meta_optimizer.reset_lstm(q2_func, keep_states=keep_states)
                    loss_sum = 0
                    # prev_loss = torch.zeros(1)

                else:
                    forward_steps += 1

                loss = q2_func.f_at_xt()
                loss.backward()
                # feed the RNN with the gradient of the error surface function
                meta_q2func = meta_optimizer.meta_update(q2_func)
                loss = meta_q2func.f_at_xt()

                # print("x: %.3f / Loss quad %.3f" % (x_t.data.numpy()[0], loss_quad.data.numpy()[0]))
                loss_sum = loss_sum + loss
                if forward_steps == args.truncated_bptt_step or k == args.optimizer_steps - 1:
                    # print("INFO@step %d(%d) - Backprop for LSTM" % (k, forward_steps))
                    # Update the parameters of the meta optimizer
                    meta_optimizer.zero_grad()
                    loss_sum.backward()
                    for param in meta_optimizer.parameters():
                        param.grad.data.clamp_(-1, 1)
                    optimizer.step()
                    meta_q2func.parameter.grad.data.zero_()
            # end of for-loop for a specific function to be optimized, register final loss for quadratic
            loss_diff += initial_loss.data[0] - loss.data[0]
            final_loss += loss.data[0]
        # end of an epoch, print summary
        loss_diff *= 1./args.functions_per_epoch
        final_loss *= 1./args.functions_per_epoch
        end_epoch = time.time()
        print("Epoch-info: elapsed time %.2f seconds" % (end_epoch - start_epoch))
        if epoch % args.print_freq == 0 or epoch + 1 == args.max_epoch:
            validate_optimizer(meta_optimizer)
            print("Epoch: {}, average final loss {}, average diff (initial-last) loss/function: {}".format(epoch,
                                                                                                   final_loss[0],
                                                                                                   loss_diff[0]))

if __name__ == "__main__":
    main()
