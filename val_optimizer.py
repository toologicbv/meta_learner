import numpy as np
import os

import torch
import torch.optim as optim
from torch.autograd import Variable
from utils.quadratic import Quadratic, Quadratic2D

from utils.config import config
from utils.utils import softmax
from utils.probs import ConditionalTimeStepDist


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
        val_set = [Quadratic2D(use_cuda=exper.args.cuda)]
        plot_idx = [0]
    else:
        plot_idx = [(i + 1) * (len(val_set) // num_of_plots) - 1 for i in range(num_of_plots)]
    num_of_funcs = len(val_set)

    meta_logger.info("INFO - Epoch {}: Validating meta-learner with {} functions".format(exper.epoch, num_of_funcs))
    total_loss = 0
    total_param_loss = 0
    total_act_loss = 0
    for f, q_func in enumerate(val_set):
        if verbose and f in plot_idx:
            meta_logger.info("******* {}-th validation function *******".format(f + 1))
        if exper.args.q2D:
            q_func.use_cuda = exper.args.cuda
            q_func.reset()
            if verbose and f in plot_idx:
                meta_logger.info("\tStart-value parameters ({:.2f},{:.2f})".format(
                    q_func.parameter[0].data.numpy()[0], q_func.parameter[1].data.numpy()[0]))
        else:
            q_func.use_cuda = exper.args.cuda
            q_func.reset()
            if verbose and f in plot_idx:
                meta_logger.info("\tStart-value parameter {}, true min {}".format(q_func.parameter.squeeze().data[0],
                                                          q_func.true_opt.squeeze().data[0]))

        if exper.args.learner == 'manual':
            state_dict = meta_learner.state_dict()
            meta_learner = optim.Adam([q_func.parameter], lr=STD_OPT_LR)
            meta_learner.load_state_dict(state_dict)
        elif exper.args.learner == "act":
            meta_learner.reset_final_loss()

        qt_weights = []
        for i in range(steps):
            # Keep states for truncated BPTT
            if i > exper.args.truncated_bptt_step - 1:
                keep_states = True
            else:
                keep_states = False
            if i % exper.args.truncated_bptt_step == 0 and not exper.args.learner == 'manual':
                meta_learner.reset_lstm(q_func, keep_states=keep_states)

            if exper.args.loss_type == "EVAL":
                loss = q_func.f_at_xt(hist=True)
            else:
                # we need to compute two losses, in order to plot the gradient path
                _ = q_func.f_at_xt(hist=True)
                loss = q_func.compute_error()

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
                    qt_weights.append(delta_p[1].data.squeeze().numpy()[0])
                    meta_learner.losses.append(Variable(loss.data))
                    meta_learner.q_t.append(delta_p[1])
                # Update the parameter of the function that is optimized
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
        total_param_loss += q_func.compute_error().data.squeeze().numpy()[0]
        if exper.args.learner == "act":
            # get prior probs for this number of optimizer steps
            kl_prior_dist = ConditionalTimeStepDist(steps)
            priors = Variable(torch.from_numpy(kl_prior_dist.pmfunc(np.arange(1, steps + 1))).float())
            act_loss = meta_learner.final_loss(prior_probs=priors)
            total_act_loss += act_loss.data
            # get hold of q(t|T) values and step-count frequency. will use this for plotting of various q(t|T) dists
            meta_learner.qt_hist_val[steps] += meta_learner.q_soft.data.squeeze().numpy()
            meta_learner.opt_step_hist_val[steps - 1] += 1

        if exper.args.q2D:
            if verbose and f in plot_idx:
                meta_logger.info("\tTrue parameter values ({:.2f},{:.2f})".format(q_func.true_opt[0].data.numpy()[0],
                                                                     q_func.true_opt[1].data.numpy()[0]))
                meta_logger.info("\tFinal parameter values ({:.2f},{:.2f})".format(q_func.parameter[0].data.numpy()[0],
                                                                      q_func.parameter[1].data.numpy()[0]))
                if exper.args.learner == 'act':
                    meta_logger.info("Final qt-probabilities")
                    meta_logger.info("raw:   {}".format(np.array_str(np.array(qt_weights))))
                    meta_logger.info("probs: {}".format(np.array_str(softmax(np.array(qt_weights)))))
                    meta_logger.info("losses: {}".format(np.array_str(torch.cat(
                        meta_learner.losses, 0).data.squeeze().numpy())))
        else:
            if verbose and f in plot_idx:
                meta_logger.info("\tFinal parameter values {:.2f}".format(q_func.parameter.data.numpy()[0]))
        if verbose and f in plot_idx:
            meta_logger.info("\tValidation - final error: {:.4}".format(loss_diff[0]))
        total_loss += loss_diff

    total_loss *= 1. / (f + 1)
    total_param_loss *= 1. / (f + 1)
    if exper.args.learner == "act":
        total_act_loss *= 1. / (f + 1)
        exper.val_stats["act_loss"].append(total_act_loss[0])
        # reset some variables of the meta learner otherwise the training procedures will have serious problems
        meta_learner.reset_final_loss()
    exper.val_stats["loss"].append(total_loss[0])
    exper.val_stats["param_error"].append(total_param_loss)
    meta_logger.info("INFO - Epoch {}: Final validation average loss / param-loss: {:.4}/{:.4}".format(exper.epoch,
                                                                                                       total_loss[0],
                                                                                                       total_param_loss))
