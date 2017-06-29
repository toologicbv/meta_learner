import argparse
import time
import numpy as np

import torch
from torch.autograd import Variable
from utils.regression import RegressionFunction, L2LQuadratic

from utils.config import config
from utils.utils import load_val_data, Experiment, prepare, end_run, get_model, print_flags
from utils.utils import create_logger, detailed_train_info, construct_prior_p_t_T
from utils.probs import TimeStepsDist, ConditionalTimeStepDist
from val_optimizer import validate_optimizer

"""
    TODO:
    (1) cuda seems at least to "run" but takes twice the time of cpu implementation (25 secs <-> 55 secs)
    (2) re-training does not seem to work. at a certain moment the model predicts gibberish. first thought
    this had to do with validation of model but tested without validation and result stays the same.
"""

MAX_VAL_FUNCS = 20000
# for standard optimizer which we compare to
STD_OPT_LR = 4e-1
VALID_VERBOSE = False
TRAIN_VERBOSE = False
PLOT_VALIDATION_FUNCS = False
NOISE_SIGMA = 1.
ANNEAL_LR = False
ACT_TRUNC_BPTT = False

OPTIMIZER_DICT = {'sgd': torch.optim.SGD, # Gradient Descent
                  'adadelta': torch.optim.Adadelta, # Adadelta
                  'adagrad': torch.optim.Adagrad, # Adagrad
                  'adam': torch.optim.Adam, # Adam
                  'rmsprop': torch.optim.RMSprop # RMSprop
                  }

# python train_optimizer.py --max_epoch=10 --learner=act --version=V2 --lr=4e-6 --batch_size=125 --hidden_size=40
# --functions_per_epoch=10000 --use_cuda --eval_freq=10 --optimizer_steps=100 --problem="regression"
# --fixed_horizon --optimizer_steps=100

parser = argparse.ArgumentParser(description='PyTorch Meta-learner')

parser.add_argument('--x_dim', type=int, default=10, metavar='N',
                    help='dimensionality of the regression variable x (default: 10)')
parser.add_argument('--lr', type=float, default=1e-5, metavar='N',
                    help='default learning rate for optimizer (default: 1e-5)')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='number of functions per batch (default: 20)')
parser.add_argument('--optimizer_steps', type=int, default=10, metavar='N',
                    help='number of meta optimizer steps (default: 10)')
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
    # get distribution P(T) over possible number of total timesteps
    if args.learner == "act":
        if args.version[0:2] not in ['V1', 'V2']:
            raise ValueError("Version {} currently not supported (only V1.x and V2.x)".format(args.version))
        pt_dist = TimeStepsDist(config.T, config.continue_prob)
        if args.fixed_horizon:
            exper.avg_num_opt_steps = args.optimizer_steps
        else:
            exper.avg_num_opt_steps = pt_dist.mean
    else:
        exper.avg_num_opt_steps = args.optimizer_steps
        if args.learner == 'meta' and args.version[0:2] == 'V2':
            pt_dist = TimeStepsDist(config.T, config.continue_prob)
            exper.avg_num_opt_steps = pt_dist.mean
    # prepare the experiment
    exper.output_dir = prepare(prcs_args=args, exper=exper)
    # get our logger (one to std-out and one to file)
    meta_logger = create_logger(exper, file_handler=True)
    # print the argument flags
    print_flags(args, meta_logger)
    # load the validation functions
    if args.problem == "quadratic":
        val_funcs = load_val_data(size=MAX_VAL_FUNCS, n_samples= args.x_samples, noise_sigma=NOISE_SIGMA, dim=args.x_dim,
                                  logger=meta_logger, file_name="10d_quadratic_val_funcs_15000.dll")
    else:
        val_funcs = load_val_data(size=MAX_VAL_FUNCS, n_samples=args.x_samples, noise_sigma=NOISE_SIGMA, dim=args.x_dim,
                                  logger=meta_logger)
    # exper.val_funcs = val_funcs
    lr = args.lr
    if not args.learner == 'manual':
        # Important switch, use meta optimizer (LSTM) which will be trained
        meta_optimizer = get_model(exper, args.x_dim, retrain=args.retrain, logger=meta_logger)
        optimizer = OPTIMIZER_DICT[args.optimizer](meta_optimizer.parameters(), lr=lr)
    else:
        # we're using one of the standard optimizers, initialized per function below
        meta_optimizer = None
        optimizer = None
    # prepare epoch variables
    backward_ones = torch.ones(args.batch_size)
    if args.cuda:
        backward_ones = backward_ones.cuda()
    num_of_batches = args.functions_per_epoch // args.batch_size
    if args.fixed_horizon and args.learner == "act":
        prior_probs = construct_prior_p_t_T(args.optimizer_steps, config.continue_prob, args.batch_size,
                                            args.cuda)

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

        if (args.learner == "meta" and args.version == "V2") or \
                (args.learner == "act" and not args.fixed_horizon):
            avg_opt_steps = []
        else:
            # in all other cases we're working with a fixed optimization horizon H=args.optimizer_steps
            avg_opt_steps = [args.optimizer_steps]
            optimizer_steps = args.optimizer_steps

        for i in range(num_of_batches):

            if args.problem == "quadratic":
                reg_funcs = L2LQuadratic(batch_size=args.batch_size, num_dims=args.x_dim, stddev=0.01,
                                         use_cuda=args.cuda)

            elif args.problem == "regression":
                reg_funcs = RegressionFunction(n_funcs=args.batch_size, n_samples=args.x_samples,
                                               stddev=NOISE_SIGMA, x_dim=args.x_dim,
                                               use_cuda=args.cuda)
            # if we're using a standard optimizer
            if args.learner == 'manual':
                meta_optimizer = OPTIMIZER_DICT[args.optimizer]([reg_funcs.params], lr=STD_OPT_LR)

            # counter that we keep in order to enable BPTT
            forward_steps = 0
            # determine the number of optimization steps for this batch
            if args.learner == 'meta' and args.version[0:2] == 'V2':
                optimizer_steps = pt_dist.rvs(n=1)[0]
                avg_opt_steps.append(optimizer_steps)
            elif args.learner == 'act' and not args.fixed_horizon:
                # sample T - the number of timesteps - from our PMF (note prob to continue is set in config object)
                # add one to choice because we actually want values between [1, config.T]
                optimizer_steps = pt_dist.rvs(n=1)[0]
                prior_probs = construct_prior_p_t_T(optimizer_steps, config.continue_prob, args.batch_size,
                                                        args.cuda)
                avg_opt_steps.append(optimizer_steps)

            # the q-parameter for the ACT model, initialize
            qt_param = Variable(torch.zeros(args.batch_size, 1))
            if args.cuda:
                qt_param = qt_param.cuda()

            # outer loop with the optimization steps
            for k in range(optimizer_steps):

                if args.learner == 'meta' or (args.learner == "act" and ACT_TRUNC_BPTT):
                    # meta model uses truncated BPTT
                    # Keep states for truncated BPTT
                    if k > args.truncated_bptt_step - 1:
                        keep_states = True
                    else:
                        keep_states = False
                    if k % args.truncated_bptt_step == 0 and not args.learner == 'manual':
                        # meta_logger.debug("DEBUG@step %d - Resetting LSTM" % k)
                        forward_steps = 1
                        meta_optimizer.reset_lstm(keep_states=keep_states)
                        # kind of fake reset, the actual value of the function parameters are NOT changed, only
                        # the pytorch Variable, in order to prevent the .backward() function to go beyond the truncated
                        # BPTT steps
                        reg_funcs.reset_params()
                        loss_sum = 0
                    else:
                        forward_steps += 1
                elif args.learner == 'act' and not ACT_TRUNC_BPTT and k == 0:
                    # ACT model (if not using truncated BPTT) the LSTM hidden states will be only initialized
                    # for the first optimization step
                    forward_steps = 1
                    # initialize LSTM
                    meta_optimizer.reset_lstm(keep_states=False)
                    reg_funcs.reset_params()
                    loss_sum = 0

                if args.problem == "quadratic":
                    loss = reg_funcs.compute_loss(average=False)
                else:
                    loss = reg_funcs.compute_neg_ll(average_over_funcs=False, size_average=False)
                # compute gradients of optimizee which will need for the meta-learner
                loss.backward(backward_ones)
                # print("Sum gradients ", torch.sum(reg_funcs.params.grad.data))
                # feed the RNN with the gradient of the error surface function
                if args.learner == 'meta':
                    delta_param = meta_optimizer.meta_update(reg_funcs)
                    par_new = reg_funcs.params - delta_param
                    if args.problem == "quadratic":
                        loss_step = reg_funcs.compute_loss(average=True, params=par_new)
                        meta_optimizer.losses.append(Variable(loss_step.data))
                    else:
                        # Regression
                        loss_step = meta_optimizer.step_loss(reg_funcs, par_new, average_batch=True)

                    reg_funcs.set_parameters(par_new)
                    loss_sum = loss_sum + loss_step
                # ACT model processing
                elif args.learner == 'act':
                    delta_param, delta_qt = meta_optimizer.meta_update(reg_funcs)
                    par_new = reg_funcs.params - delta_param
                    qt_param = qt_param + delta_qt
                    if args.problem == "quadratic":
                        loss_step = reg_funcs.compute_loss(average=False, params=par_new)
                        meta_optimizer.losses.append(loss_step)
                        loss_step = 1/float(reg_funcs.num_of_funcs) * torch.sum(loss_step)
                    else:
                        # Regression
                        loss_step = meta_optimizer.step_loss(reg_funcs, par_new, average_batch=True)
                    meta_optimizer.q_t.append(qt_param)
                    loss_sum = loss_sum + loss_step
                    reg_funcs.set_parameters(par_new)
                else:
                    # we're just using one of the pre-delivered optimizers, update function parameters
                    meta_optimizer.step()
                    # compute loss after update
                    loss_step = reg_funcs.compute_neg_ll(average_over_funcs=False, size_average=False)

                reg_funcs.params.grad.data.zero_()

                if forward_steps == args.truncated_bptt_step or k == optimizer_steps - 1:
                    # meta_logger.info("BPTT at {}".format(k + 1))
                    if args.learner == 'meta' or (args.learner == 'act' and args.version[0:2] == "V1"):
                        loss_sum.backward()
                        optimizer.step()
                        meta_optimizer.zero_grad()
                    # this is the part where we tried to implement truncated BPTT for the ACTV2 model
                    # is only used if ACT_TRUNC_BPTT set to TRUE
                    elif args.learner == 'act' and args.version[0:2] == "V2" and ACT_TRUNC_BPTT:
                        T = k+1
                        prior_probs = construct_prior_p_t_T(T, config.continue_prob, args.batch_size,
                                                            args.cuda)
                        if args.cuda:
                            prior_probs = prior_probs.cuda()
                        # we need to expand the prior probs to the size of the batch
                        prior_probs = prior_probs.expand(args.batch_size, prior_probs.size(0))
                        act_loss = meta_optimizer.final_loss(prior_probs, run_type='train')
                        if k == optimizer_steps - 1:
                            act_loss.backward()
                            final_act_loss += act_loss.data
                        else:
                            # intermediate back-propagation
                            act_loss.backward(retain_variables=True)
                        optimizer.step()
                        meta_optimizer.zero_grad()
            # END of iterative function optimization. Compute final losses and probabilities

            # compute the final loss error for this function between last loss calculated and function min-value
            error = loss_step.data

            # back-propagate ACT loss that was accumulated during optimization steps
            if args.learner == 'act' and not ACT_TRUNC_BPTT:
                # processing ACT loss
                act_loss = meta_optimizer.final_loss(prior_probs, run_type='train')
                act_loss.backward()
                optimizer.step()
                final_act_loss += act_loss.data
                # set grads of meta_optimizer to zero after update parameters
                meta_optimizer.zero_grad()

            # Does it work? if necessary set TRAIN_VERBOSE to True
            if i % 20 == 0 and i != 0 and TRAIN_VERBOSE:
                detailed_train_info(meta_logger, reg_funcs, 0, args, meta_optimizer, i, optimizer_steps, error[0])

            if args.learner == "act":
                meta_optimizer.reset_final_loss()
            elif exper.args.learner == 'meta':
                meta_optimizer.reset_losses()
            # END OF BATCH: FUNCTION OPTIMIZATION

            final_loss += error
            param_loss += reg_funcs.param_error(average=True).data
        # END OF EPOCH, calculate average final loss/error and print summary
        # we computed the average loss per function in the batch! and added those losses to the final loss
        # therefore we only need to divide through the number of batches to end up with the average loss per function
        final_loss *= 1./float(num_of_batches)
        param_loss *= 1./float(num_of_batches)
        final_act_loss *= 1./float(num_of_batches)
        end_epoch = time.time()

        meta_logger.info("Epoch: {}, elapsed time {:.2f} seconds: average final loss {:.4f} / param-loss {:.4f}".format(
              epoch+1, (end_epoch - start_epoch), final_loss[0], param_loss[0]))
        if args.learner == 'act':
            meta_logger.info("Epoch: {}, ACT - average final act_loss {:.4f}".format(epoch+1, final_act_loss[0]))
            avg_opt_steps = int(np.mean(np.array(avg_opt_steps)))
            meta_logger.debug("Epoch: {}, Average number of optimization steps {}".format(epoch+1, avg_opt_steps))
        if args.learner == 'meta' and args.version[0:2] == "V2":
            avg_opt_steps = int(np.mean(np.array(avg_opt_steps)))
            meta_logger.debug("Epoch: {}, Average number of optimization steps {}".format(epoch + 1, avg_opt_steps))

        exper.epoch_stats["loss"].append(final_loss[0])
        exper.epoch_stats["param_error"].append(param_loss[0])
        if args.learner == 'act':
            exper.epoch_stats["act_loss"].append(final_act_loss[0])
        # if applicable, VALIDATE model performance
        if exper.epoch % args.eval_freq == 0 or epoch + 1 == args.max_epoch:

            if args.learner == 'manual':
                # the manual (e.g. SGD, Adam will be validated using full number of optimization steps
                opt_steps = args.optimizer_steps
            else:
                opt_steps = config.max_val_opt_steps

            validate_optimizer(meta_optimizer, exper, val_set=val_funcs, meta_logger=meta_logger,
                               verbose=VALID_VERBOSE,
                               plot_func=PLOT_VALIDATION_FUNCS,
                               max_steps=opt_steps,
                               num_of_plots=config.num_val_plots,
                               save_qt_prob_funcs=True if epoch + 1 == args.max_epoch else False,
                               save_model=True)
        # per epoch collect the statistics w.r.t q(t|T) distribution for training and validation
        if args.learner == 'act':

            exper.epoch_stats['qt_hist'][exper.epoch] = meta_optimizer.qt_hist
            exper.epoch_stats['opt_step_hist'][exper.epoch] = meta_optimizer.opt_step_hist

            meta_optimizer.init_qt_statistics(exper.config)

    end_run(exper, meta_optimizer, validation=False, on_server=args.on_server)

if __name__ == "__main__":
    main()
