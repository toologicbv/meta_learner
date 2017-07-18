import argparse
import time
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn

from utils.config import config
from utils.utils import load_val_data, Experiment, prepare, end_run, get_model, print_flags
from utils.utils import create_logger, detailed_train_info, construct_prior_p_t_T, generate_fixed_weights
from utils.utils import get_batch_functions, get_func_loss
from utils.probs import TimeStepsDist
from val_optimizer import validate_optimizer

"""
    TODO:
    (1) cuda seems at least to "run" but takes twice the time of cpu implementation (25 secs <-> 55 secs)
    (2) re-training does not seem to work. at a certain moment the model predicts gibberish. first thought
    this had to do with validation of model but tested without validation and result stays the same.
"""

VALIDATE = True
# for standard optimizer which we compare to
STD_OPT_LR = 4e-1
VALID_VERBOSE = False
TRAIN_VERBOSE = False
PLOT_VALIDATION_FUNCS = False
ANNEAL_LR = False

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
    # get distribution P(T) over possible number of total timesteps
    if args.learner == "act":
        if exper.args.version[0:2] not in ['V1', 'V2']:
            raise ValueError("Version {} currently not supported (only V1.x and V2.x)".format(args.version))
        pt_dist = TimeStepsDist(T=config.T, q_prob=config.pT_shape_param)
        if args.fixed_horizon:
            exper.avg_num_opt_steps = args.optimizer_steps
        else:
            exper.avg_num_opt_steps = pt_dist.mean
    else:
        exper.avg_num_opt_steps = args.optimizer_steps
        if args.learner == 'meta' and args.version[0:2] == 'V2':
            # Note, we choose here an absolute limit of the horizon, set in the config-file
            pt_dist = TimeStepsDist(T=config.T, q_prob=config.pT_shape_param)
            exper.avg_num_opt_steps = pt_dist.mean

    # prepare the experiment
    exper = prepare(exper=exper)
    # get our logger (one to std-out and one to file)
    meta_logger = create_logger(exper, file_handler=True)
    # print the argument flags
    print_flags(exper, meta_logger)

    # load the validation functions
    if VALIDATE:
        val_funcs = load_val_data(num_of_funcs=exper.config.num_val_funcs, n_samples=exper.args.x_samples,
                                  stddev=exper.config.stddev, dim=exper.args.x_dim, logger=meta_logger,
                                  exper=exper)
    else:
        val_funcs = None

    lr = exper.args.lr
    if not exper.args.learner == 'manual':
        meta_optimizer = get_model(exper, exper.args.x_dim, retrain=exper.args.retrain, logger=meta_logger)
        optimizer = OPTIMIZER_DICT[exper.args.optimizer](meta_optimizer.parameters(), lr=lr)
        fixed_weights = generate_fixed_weights(exper, meta_logger)
    else:
        # we're using one of the standard optimizers, initialized per function below
        meta_optimizer = None
        optimizer = None
        fixed_weights = None
    # prepare epoch variables
    backward_ones = torch.ones(exper.args.batch_size)
    if exper.args.cuda:
        backward_ones = backward_ones.cuda()
    num_of_batches = exper.args.functions_per_epoch // exper.args.batch_size
    if exper.args.fixed_horizon and exper.args.learner == "act":
        prior_probs = construct_prior_p_t_T(exper.args.optimizer_steps, config.ptT_shape_param, exper.args.batch_size,
                                            exper.args.cuda)

    for epoch in range(exper.args.max_epoch):
        exper.epoch += 1
        start_epoch = time.time()
        final_loss = 0.0
        final_act_loss = 0.
        param_loss = 0.
        total_loss_steps = 0.
        loss_optimizer = 0.
        diff_min = 0.
        # in each epoch we optimize args.functions_per_epoch functions in total, packaged in batches of args.batch_size
        # and therefore ideally functions_per_epoch should be a multiple of batch_size
        # ALSO NOTE:
        #       for ACT models we sample for each batch the horizon T aka the number of optimization steps
        #       THEREFORE we should make sure that we have lots of batches per epoch e.g. 5000 functions and
        #       batch size of 50

        if (exper.args.learner == "meta" and exper.args.version == "V2") or \
                (exper.args.learner == "act" and not exper.args.fixed_horizon):
            avg_opt_steps = []
        else:
            # in all other cases we're working with a fixed optimization horizon H=exper.args.optimizer_steps
            avg_opt_steps = [exper.args.optimizer_steps]
            optimizer_steps = exper.args.optimizer_steps

        for i in range(num_of_batches):

            reg_funcs = get_batch_functions(exper, exper.config.stddev)
            func_is_nn_module = nn.Module in reg_funcs.__class__.__bases__
            # if we're using a standard optimizer
            if exper.args.learner == 'manual':
                meta_optimizer = OPTIMIZER_DICT[exper.args.optimizer]([reg_funcs.params], lr=STD_OPT_LR)

            # counter that we keep in order to enable BPTT
            forward_steps = 0
            # determine the number of optimization steps for this batch
            if exper.args.learner == 'meta' and exper.args.version[0:2] == 'V2':
                optimizer_steps = pt_dist.rvs(n=1)[0]
                avg_opt_steps.append(optimizer_steps)
            elif exper.args.learner == 'act' and not exper.args.fixed_horizon:
                # sample T - the number of timesteps - from our PMF (note prob to continue is set in config object)
                # add one to choice because we actually want values between [1, config.T]
                optimizer_steps = pt_dist.rvs(n=1)[0]
                prior_probs = construct_prior_p_t_T(optimizer_steps, config.ptT_shape_param, exper.args.batch_size,
                                                        exper.args.cuda)
                avg_opt_steps.append(optimizer_steps)

            # the q-parameter for the ACT model, initialize
            qt_param = Variable(torch.zeros(exper.args.batch_size, 1))
            if exper.args.cuda:
                qt_param = qt_param.cuda()

            # outer loop with the optimization steps
            # meta_logger.info("Batch size {}".format(reg_funcs.num_of_funcs))
            for k in range(optimizer_steps):

                if exper.args.learner == 'meta':
                    # meta model uses truncated BPTT, Keep states for truncated BPTT
                    if k > exper.args.truncated_bptt_step - 1:
                        keep_states = True
                    else:
                        keep_states = False
                    if k % exper.args.truncated_bptt_step == 0 and not exper.args.learner == 'manual':
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
                elif exper.args.learner == 'act' and k == 0:
                    # ACT model: the LSTM hidden states will be only initialized
                    # for the first optimization step
                    forward_steps = 1
                    # initialize LSTM
                    meta_optimizer.reset_lstm(keep_states=False)
                    reg_funcs.reset_params()
                    loss_sum = 0
                loss = get_func_loss(exper, reg_funcs, average=False)
                # compute gradients of optimizee which will need for the meta-learner
                loss.backward(backward_ones)
                total_loss_steps += torch.mean(loss, 0).data.cpu().squeeze().numpy()[0].astype(float)
                # new_version ("observed improvement")
                # if exper.args.learner == 'meta' and k == 0:
                #    meta_optimizer.losses.append(Variable(torch.mean(loss, 0).data.squeeze()))

                # meta_logger.info("{}/{} Sum optimizee gradients {:.3f}".format(
                #    i, k, torch.sum(reg_funcs.params.grad.data)))
                # feed the RNN with the gradient of the error surface function
                if exper.args.learner == 'meta':
                    delta_param = meta_optimizer.meta_update(reg_funcs)
                    if exper.args.problem == "quadratic":
                        par_new = reg_funcs.params - delta_param
                        loss_step = reg_funcs.compute_loss(average=True, params=par_new)
                        meta_optimizer.losses.append(Variable(loss_step.data))
                    elif exper.args.problem == "regression":
                        # Regression
                        par_new = reg_funcs.params - delta_param
                        loss_step = meta_optimizer.step_loss(reg_funcs, par_new, average_batch=True)
                    elif exper.args.problem == "rosenbrock":
                        par_new = reg_funcs.get_flat_params() + delta_param
                        loss_step = reg_funcs.evaluate(parameters=par_new, average=True)
                        meta_optimizer.losses.append(Variable(loss_step.data.unsqueeze(1)))

                    reg_funcs.set_parameters(par_new)
                    if exper.args.version[0:2] == "V3":
                        loss_sum = loss_sum + torch.mul(fixed_weights[k], loss_step)
                    else:
                        # new_version (observed improvement)
                        # min_f = torch.min(torch.cat(meta_optimizer.losses[0:k+1], 0))
                        # observed_imp = loss_step - min_f
                        # if i == 79:
                        #    print(min_f.data.cpu().numpy()[0], loss_step.data.cpu().numpy()[0])
                        #    meta_logger.info("Step {} - OI {:.3f}".format(k,
                        #                                              observed_imp.data.cpu().squeeze().numpy()[0]))
                        # loss_sum += observed_imp
                        loss_sum = loss_sum + loss_step
                # ACT model processing
                elif exper.args.learner == 'act':
                    delta_param, delta_qt = meta_optimizer.meta_update(reg_funcs)
                    par_new = reg_funcs.params - delta_param
                    qt_param = qt_param + delta_qt
                    if exper.args.problem == "quadratic":
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

                # set gradients of optimizee to zero again
                if func_is_nn_module:
                    reg_funcs.zero_grad()
                else:
                    reg_funcs.params.grad.data.zero_()

                if forward_steps == exper.args.truncated_bptt_step or k == optimizer_steps - 1:
                    # meta_logger.info("BPTT at {}".format(k + 1))
                    if exper.args.learner == 'meta' or (exper.args.learner == 'act' and exper.args.version[0:2] == "V1"):
                        # meta_logger.info("{}/{} Sum loss {:.3f}".format(i, k,
                        # loss_sum.data.cpu().squeeze().numpy()[0]))
                        loss_sum.backward()
                        optimizer.step()
                        meta_optimizer.zero_grad()
                        # Slightly sloppy. Actually for the ACTV1 model we only register the ACT loss as the
                        # so called optimizer-loss. But actually ACTV1 is using both losses
                        if exper.args.learner == 'meta':
                            loss_optimizer += loss_sum.data

            # END of iterative function optimization. Compute final losses and probabilities
            # compute the final loss error for this function between last loss calculated and function min-value
            error = loss_step.data
            diff_min += (loss_step - reg_funcs.true_minimum_nll.expand_as(loss_step)).data.cpu().squeeze().numpy()[0].astype(float)
            total_loss_steps += loss_step.data.cpu().squeeze().numpy()[0]
            # back-propagate ACT loss that was accumulated during optimization steps
            if exper.args.learner == 'act':
                # processing ACT loss
                act_loss = meta_optimizer.final_loss(prior_probs, run_type='train')
                act_loss.backward()
                optimizer.step()
                final_act_loss += act_loss.data
                # set grads of meta_optimizer to zero after update parameters
                meta_optimizer.zero_grad()
                loss_optimizer += act_loss.data
            # Does it work? if necessary set TRAIN_VERBOSE to True
            if i % 20 == 0 and i != 0 and TRAIN_VERBOSE:
                detailed_train_info(meta_logger, reg_funcs, 0, args, meta_optimizer, i, optimizer_steps, error[0])

            if exper.args.learner == "act":
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
        total_loss_steps *= 1./float(num_of_batches)
        loss_optimizer *= 1./float(num_of_batches)

        end_epoch = time.time()

        meta_logger.info("Epoch: {}, elapsed time {:.2f} seconds: avg optimizer loss {:.4f} / "
                         "avg total loss (over time-steps) {:.4f} /"
                         " avg final step loss {:.4f} / final-true_min {:.4f}".format(epoch+1,
                                                                                      (end_epoch - start_epoch),
                         loss_optimizer[0], total_loss_steps, final_loss[0], diff_min))
        if exper.args.learner == 'act':
            meta_logger.info("Epoch: {}, ACT - average final act_loss {:.4f}".format(epoch+1, final_act_loss[0]))
            avg_opt_steps = int(np.mean(np.array(avg_opt_steps)))
            meta_logger.debug("Epoch: {}, Average number of optimization steps {}".format(epoch+1, avg_opt_steps))
        if exper.args.learner == 'meta' and exper.args.version[0:2] == "V2":
            avg_opt_steps = int(np.mean(np.array(avg_opt_steps)))
            meta_logger.debug("Epoch: {}, Average number of optimization steps {}".format(epoch + 1, avg_opt_steps))

        exper.epoch_stats["loss"].append(total_loss_steps)
        exper.epoch_stats["param_error"].append(param_loss[0])
        if exper.args.learner == 'act':
            exper.epoch_stats["opt_loss"].append(final_act_loss[0])
        elif exper.args.learner == 'meta':
            exper.epoch_stats["opt_loss"].append(loss_optimizer[0])
        # if applicable, VALIDATE model performance
        if exper.epoch % exper.args.eval_freq == 0 or epoch + 1 == exper.args.max_epoch:

            if exper.args.learner == 'manual':
                # the manual (e.g. SGD, Adam will be validated using full number of optimization steps
                opt_steps = exper.args.optimizer_steps
            else:
                opt_steps = config.max_val_opt_steps

            validate_optimizer(meta_optimizer, exper, val_set=val_funcs, meta_logger=meta_logger,
                               verbose=VALID_VERBOSE,
                               plot_func=PLOT_VALIDATION_FUNCS,
                               max_steps=opt_steps,
                               num_of_plots=config.num_val_plots,
                               save_qt_prob_funcs=True if epoch + 1 == exper.args.max_epoch else False,
                               save_model=True)
        # per epoch collect the statistics w.r.t q(t|T) distribution for training and validation
        if exper.args.learner == 'act':

            exper.epoch_stats['qt_hist'][exper.epoch] = meta_optimizer.qt_hist
            exper.epoch_stats['opt_step_hist'][exper.epoch] = meta_optimizer.opt_step_hist

            meta_optimizer.init_qt_statistics(exper.config)
        if hasattr(meta_optimizer, "epochs_trained"):
            meta_optimizer.epochs_trained += 1

    end_run(exper, meta_optimizer, validation=True, on_server=exper.args.on_server)

if __name__ == "__main__":
    main()
