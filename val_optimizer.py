import numpy as np
import os

import torch
import torch.optim as optim
from torch.autograd import Variable
from utils.regression import RegressionFunction

from utils.config import config
from utils.utils import softmax, stop_computing
from utils.probs import ConditionalTimeStepDist


def validate_optimizer(meta_learner, exper, meta_logger, val_set=None, max_steps=6, verbose=True, plot_func=False,
                       num_of_plots=3, save_plot=True, show_plot=False):
    global STD_OPT_LR
    # we will probably call this procedure later in another context (to evaluate meta-learners)
    # so make sure the globals exist.
    if 'STD_OPT_LR' not in globals():
        meta_logger.debug("create global")
        STD_OPT_LR = 4e-1
    # initialize stats arrays
    exper.val_stats["step_losses"][exper.epoch] = np.zeros(config.max_val_opt_steps + 1)
    exper.val_stats["step_param_losses"][exper.epoch] = np.zeros(config.max_val_opt_steps + 1)

    meta_logger.info("---------------------------------------------------------------------------------------")
    if val_set is None:
        # if no validation set is provided just use one random generated q-function to run the validation
        val_set = [RegressionFunction(n_funcs=10000, n_samples=100, noise_sigma=2.5, poly_degree=3)]
        plot_idx = [0]
    else:
        plot_idx = [(i + 1) * (val_set.num_of_funcs // num_of_plots) - 1 for i in range(num_of_plots)]

    meta_logger.info("INFO - Epoch {}: Validating model {} with {} functions".format(exper.epoch, exper.args.model,
                                                                                     val_set.num_of_funcs))
    total_loss = 0
    total_param_loss = 0
    total_act_loss = 0
    if exper.args.cuda:
        val_set.enable_cuda()
    val_set.reset()
    if verbose:
        meta_logger.info("\tStart-value parameters {}".format(np.array_str(val_set.params.data.numpy()[np.array(plot_idx)])))

    if exper.args.learner == 'manual':
        state_dict = meta_learner.state_dict()
        meta_learner = optim.Adam([val_set.params], lr=STD_OPT_LR)
        meta_learner.load_state_dict(state_dict)
    elif exper.args.learner == "act":
        meta_learner.reset_final_loss()

    qt_weights = []
    do_stop = np.zeros(val_set.num_of_funcs, dtype=bool)  # initialize to False for all functions
    opt_steps = np.zeros(val_set.num_of_funcs)  # initialize to 0 for all functions
    # for the act q-probabilities, will plot them on the figure for later inspection
    str_q_probs = None
    col_losses = []
    col_param_losses = []
    qt_param = Variable(torch.zeros(val_set.num_of_funcs, 1))
    for i in range(max_steps):

        if i % exper.args.truncated_bptt_step == 0 and not exper.args.learner == 'manual':
            meta_learner.reset_lstm(keep_states=False)

        loss = val_set.compute_loss(average=True)  # (hist=False if do_stop else True)
        param_loss = val_set.param_error().data.numpy()[0].astype(float)

        if verbose and not exper.args.learner == 'manual' and i % 2 == 0:
            for f_idx in plot_idx:
                meta_logger.info("\tStep {}: current loss {:.4f}".format(str(i+1), loss.squeeze().data[f_idx]))
        loss.backward(torch.ones(val_set.num_of_funcs))

        if not exper.args.learner == 'manual':
            delta_p = meta_learner.forward(val_set.params.grad)
            if exper.args.learner == 'meta':
                # gradient descent
                par_new = val_set.params - delta_p
                loss = meta_learner.step_loss(val_set, par_new, average=True)

            elif exper.args.learner == 'act':
                # in this case forward returns a tuple (parm_delta, qt)
                par_new = val_set.params - delta_p[0]
                qt_param = qt_param + delta_p[1]
                qt_weights.append(qt_param.data.numpy().astype(float))
                loss = meta_learner.step_loss(val_set, par_new, average=True)
                meta_learner.q_t.append(qt_param)
                # we're currently not breaking out of the loop when do_stop is true, therefore we
                # need this extra do_stop condition here in order not to compute it again
                if len(qt_weights) >= 2:
                    q_logits = np.concatenate(qt_weights, 1)
                    q_probs = softmax(np.array(q_logits))
                    do_stop = stop_computing(q_probs, threshold=config.qt_threshold)

            # Update the parameter of the function that is optimized
            val_set.set_parameters(par_new)

        else:
            meta_learner.step()

        # collected the loss and parameter-error
        col_losses.append(np.around(loss.data.squeeze().numpy()[0].astype(float), decimals=2))
        col_param_losses.append(param_loss)
        val_set.params.grad.data.zero_()
        # increase the opt steps variable per function in case do_stop entry is False
        opt_steps += np.logical_not(do_stop)
        """
            ************ END OF A VALIDATION OPTIMIZATION  ******************

        """

    # make another step to register final loss

    loss = 1/float(val_set.num_of_funcs) * torch.sum(val_set.compute_loss(average=True))
    param_loss = val_set.param_error().data.numpy()[0].astype(float)
    # add to total loss
    total_param_loss += param_loss
    total_loss += loss.data.numpy()[0]

    col_losses.append(np.around(loss.data.squeeze().numpy()[0].astype(float), decimals=3))
    col_param_losses.append(param_loss)

    np_losses = np.array(col_losses)
    np_param_losses = np.array(col_param_losses)
    # concatenate losses into a string for plotting the gradient path
    if "step_losses" in exper.val_stats.keys():
        exper.val_stats["step_losses"][exper.epoch] += np_losses
    if "step_param_losses" in exper.val_stats.keys():
        exper.val_stats["step_param_losses"][exper.epoch] += np_param_losses

    str_losses = np.array_str(np_losses, precision=3, suppress_small=True)
    if exper.args.learner == "act":
        # get prior probs for this number of optimizer steps
        # TODO currently we set T=max_steps because we are not stopping at the optimal step!!!
        kl_prior_dist = ConditionalTimeStepDist(T=max_steps, q_prob=config.continue_prob)
        # TODO again max_steps need to be adjusted later here when we really stop!!!
        priors = Variable(torch.from_numpy(kl_prior_dist.pmfunc(np.arange(1, max_steps + 1))).float())
        priors = priors.expand(val_set.num_of_funcs, max_steps)
        total_act_loss = meta_learner.final_loss(prior_probs=priors, run_type='val').data
        str_q_probs = np.array_str(np.around(softmax(np.array(qt_weights)), decimals=5))
        exper.val_stats["qt_dist"][exper.epoch] = np.mean(q_probs, 0)

    if verbose:
        for f in plot_idx:
            meta_logger.info("\tf{}: true parameter values: {})".format(str(i+1),
                                                                        np.array_str(val_set.true_params.data.numpy()[f, :])))
            meta_logger.info("\tf{}: final parameter values ({:.2f},{:.2f})".format(str(i+1),
                                                                        np.array_str(val_set.params.data.numpy()[f, :])))
        if exper.args.learner == 'act':
            meta_logger.info("Final qt-probabilities")
            meta_logger.info("raw:   {}".format(np.array_str(np.array(qt_weights))))
            meta_logger.info("probs: {}".format(str_q_probs))
            meta_logger.info("losses: {}".format(str_losses))

    # only plot function in certain cases, last condition...exceptionally if we found one in 2 opt-steps
    if plot_func:
        for f in plot_idx:
            # val_set.plot_func(f_idx=f, do_save=save_plot, show=show_plot, exper=exper, steps=[0, 1, 2, 3, 4])
            val_set.plot_opt_steps(f_idx=f, do_save=save_plot, show=show_plot, exper=exper,
                                   add_text=None)
        # q_func.plot_func(fig_name=fig_name_prefix, show=show_plot, do_save=save_plot, exper=exper,
        #                 add_text=(str_q_probs, str_losses))
    if exper.args.learner == 'act':
        # reset some variables of the meta learner otherwise the training procedures will have serious problems
        meta_learner.reset_final_loss()
    elif exper.args.learner == 'meta':
        meta_learner.reset_losses()

    """
        ****************** END OF VALIDATION **********************
                Register results for this validation run
    """

    exper.val_stats["step_param_losses"][exper.epoch] = np.around(exper.val_stats["step_param_losses"][exper.epoch],
                                                                  decimals=4)
    exper.val_stats["loss"].append(total_loss)
    if exper.args.learner == "act":

        exper.val_stats["act_loss"].append(total_act_loss)
        exper.val_avg_num_opt_steps = int(np.mean(opt_steps))
        meta_logger.info("INFO - Avg number of steps: {}".format(exper.val_avg_num_opt_steps))
        meta_logger.debug("{} CDF q(t) {}".format(exper.epoch, np.array_str(np.cumsum(np.mean(q_probs, 0)),
                                                                            precision=4)))
        meta_logger.info("INFO - Epoch {}: Final validation average ACT-loss: {:.4}".format(exper.epoch,
                                                                                            total_act_loss))
    exper.val_stats["param_error"].append(total_param_loss)
    meta_logger.info("INFO - Epoch {}: Final validation average loss / param-loss: {:.4}/{:.4}".format(exper.epoch,
                                                                                                       total_loss,
                                                                                                       total_param_loss))
    meta_logger.info("INFO - Epoch {}: Final step param-losses: {}".format(exper.epoch,
                     np.array_str(exper.val_stats["step_param_losses"][exper.epoch], precision=4)))

    meta_logger.info("--------------------------- End of validation --------------------------------------------")
