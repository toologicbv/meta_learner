import numpy as np
import os

import torch
import torch.optim as optim
from torch.autograd import Variable
from utils.quadratic import Quadratic, Quadratic2D

from utils.config import config
from utils.utils import softmax, stop_computing
from utils.probs import ConditionalTimeStepDist


def validate_optimizer(meta_learner, exper, meta_logger, val_set=None, steps=6, verbose=True, plot_func=False,
                       num_of_plots=3, save_plot=True, show_plot=False):
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
        do_stop = False
        opt_steps = 0
        # for the act q-probabilities, will plot them on the figure for later inspection
        str_q_probs = None
        col_losses = []
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
            col_losses.append(np.around(loss.data.squeeze().numpy()[0].astype(float), decimals=2))
            if not exper.args.learner == 'manual':
                delta_p = meta_learner.forward(q_func.parameter.grad)
                if exper.args.learner == 'meta':
                    # gradient descent
                    par_new = q_func.parameter.data - delta_p.data
                    meta_learner.losses.append(Variable(loss.data))
                elif exper.args.learner == 'act':
                    # in this case forward returns a tuple (parm_delta, qt)
                    par_new = q_func.parameter.data - delta_p[0].data
                    qt_weights.append(delta_p[1].data.squeeze().numpy()[0].astype(float))
                    meta_learner.losses.append(Variable(loss.data))
                    meta_learner.q_t.append(delta_p[1])
                    if len(qt_weights) >= 2:
                        do_stop = stop_computing(qt_weights, meta_logger)

                        meta_logger.debug("losses {}".format(np.array_str(np.array([l.data.squeeze().numpy()[0]
                                                                                    for l in meta_learner.losses]))))
                # Update the parameter of the function that is optimized
                q_func.parameter.data.copy_(par_new)
            else:
                meta_learner.step()

            q_func.parameter.grad.data.zero_()
            opt_steps = i+1

            if do_stop:
                # meta_logger.info("NOTE: stopping at {}".format(opt_steps))
                break
        """
            ************ END OF A OPTIMIZATION OF ONE SPECIFIC FUNCTION ******************
                            What follows is post processing for this f
        """
        # make another step to register final loss
        if exper.args.loss_type == "EVAL":
            loss = q_func.f_at_xt(hist=True)
        else:
            # we need to compute two losses, in order to plot the gradient path
            _ = q_func.f_at_xt(hist=True)
            loss = q_func.compute_error()
        col_losses.append(np.around(loss.data.squeeze().numpy()[0].astype(float), decimals=2))
        np_losses = np.array(col_losses)
        # concatenate losses into a string for plotting the gradient path
        if "step_losses" in exper.val_stats.keys():
            exper.val_stats["step_losses"][opt_steps] += np_losses

        str_losses = np.array_str(np_losses)
        # compute the losses (1) final loss function (2) final loss parameters (MSE)
        loss_diff = torch.abs(loss.squeeze().data - q_func.min_value.data)
        total_param_loss += q_func.compute_error().data.squeeze().numpy()[0]

        if exper.args.learner == "act":
            # get prior probs for this number of optimizer steps
            kl_prior_dist = ConditionalTimeStepDist(T=opt_steps, q_prob=config.continue_prob)
            priors = Variable(torch.from_numpy(kl_prior_dist.pmfunc(np.arange(1, opt_steps + 1))).float())
            act_loss = meta_learner.final_loss(prior_probs=priors, run_type='val')
            total_act_loss += act_loss.data
            str_q_probs = np.array_str(np.around(softmax(np.array(qt_weights)), decimals=5))

        if exper.args.q2D:
            if verbose and f in plot_idx:
                meta_logger.info("\tTrue parameter values ({:.2f},{:.2f})".format(q_func.true_opt[0].data.numpy()[0],
                                                                     q_func.true_opt[1].data.numpy()[0]))
                meta_logger.info("\tFinal parameter values ({:.2f},{:.2f})".format(q_func.parameter[0].data.numpy()[0],
                                                                      q_func.parameter[1].data.numpy()[0]))
                if exper.args.learner == 'act':
                    meta_logger.info("Final qt-probabilities")
                    meta_logger.info("raw:   {}".format(np.array_str(np.array(qt_weights))))
                    meta_logger.info("probs: {}".format(str_q_probs))
                    meta_logger.info("losses: {}".format(str_losses))
        else:
            if verbose and f in plot_idx:
                meta_logger.info("\tFinal parameter values {:.2f}".format(q_func.parameter.data.numpy()[0]))
        # only plot function in certain cases, last condition...exceptionally if we found one in 2 opt-steps
        if plot_func and f in plot_idx:
            fig_name_prefix = os.path.join(exper.output_dir, os.path.join(config.figure_path,
                                                                          str(exper.epoch) + "_f" + str(f + 1)))
            q_func.plot_func(fig_name=fig_name_prefix, show=show_plot, do_save=save_plot, exper=exper,
                             add_text=(str_q_probs, str_losses))
        if exper.args.learner == 'act':
            # reset some variables of the meta learner otherwise the training procedures will have serious problems
            meta_learner.reset_final_loss()
        elif exper.args.learner == 'meta':
            meta_learner.reset_losses()

        if verbose and f in plot_idx:
            meta_logger.info("\tValidation - final error: {:.4}".format(loss_diff[0]))
        total_loss += loss_diff

    """
        ****************** END OF VALIDATION **********************
                Register results for this validation run
    """
    total_loss *= 1. / (f + 1)
    total_param_loss *= 1. / (f + 1)
    if exper.args.learner == "act":
        total_act_loss *= 1. / (f + 1)
        exper.val_stats["act_loss"].append(total_act_loss[0])

    exper.val_stats["loss"].append(total_loss[0])
    exper.val_stats["param_error"].append(total_param_loss)
    meta_logger.info("INFO - Epoch {}: Final validation average loss / param-loss: {:.4}/{:.4}".format(exper.epoch,
                                                                                                       total_loss[0],
                                                                                                       total_param_loss))
