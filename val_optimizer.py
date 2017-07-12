import os
import time
import numpy as np

import torch
import torch.optim as optim
from torch.autograd import Variable
from utils.regression import RegressionFunction, L2LQuadratic, neg_log_likelihood_loss

from utils.config import config
from utils.utils import softmax, stop_computing, save_exper, construct_prior_p_t_T
from utils.probs import ConditionalTimeStepDist


def validate_optimizer(meta_learner, exper, meta_logger, val_set=None, max_steps=6, verbose=True, plot_func=False,
                       num_of_plots=3, save_plot=True, show_plot=False, save_qt_prob_funcs=False, save_model=False,
                       save_run=None):
    val_set = val_set
    start_validate = time.time()
    global STD_OPT_LR
    # we will probably call this procedure later in another context (to evaluate meta-learners)
    # so make sure the globals exist.
    if 'STD_OPT_LR' not in globals():
        meta_logger.debug("create global")
        STD_OPT_LR = 4e-1
    # initialize stats arrays
    exper.val_stats["step_losses"][exper.epoch] = np.zeros(exper.config.max_val_opt_steps + 1)
    exper.val_stats["step_param_losses"][exper.epoch] = np.zeros(exper.config.max_val_opt_steps + 1)

    meta_logger.info("---------------------------------------------------------------------------------------")
    if val_set is None:
        # if no validation set is provided just use one random generated q-function to run the validation
        meta_logger.info("INFO - No validation set provided, generating new functions")
        if exper.args.problem == 'regression':
            val_set = RegressionFunction(n_funcs=10000, n_samples=100, stddev=1., x_dim=10)
        else:
            val_set = L2LQuadratic(batch_size=exper.args.batch_size, num_dims=exper.args.x_dim, stddev=0.01,
                                   use_cuda=exper.args.cuda)

        plot_idx = [0]
    else:
        plot_idx = [(i + 1) * (val_set.num_of_funcs // num_of_plots) - 1 for i in range(num_of_plots)]

    meta_logger.info("INFO - Epoch {}: Validating model {} with {} functions".format(exper.epoch, exper.args.model,
                                                                                     val_set.num_of_funcs))
    total_act_loss = 0
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
    if save_qt_prob_funcs:
        exper.val_stats["loss_funcs"] = np.zeros((val_set.num_of_funcs, max_steps+1))
    # for the act q-probabilities, will plot them on the figure for later inspection
    str_q_probs = None
    col_losses = []
    col_param_losses = []
    backward_ones = torch.ones(val_set.num_of_funcs)
    if exper.args.cuda:
        backward_ones = backward_ones.cuda()
    qt_param = Variable(torch.zeros(val_set.num_of_funcs, 1))
    if exper.args.cuda:
        qt_param = qt_param.cuda()
    # Initialize hidden cell state of LSTM
    if not exper.args.learner == 'manual':
        meta_learner.reset_lstm(keep_states=False)
    for i in range(max_steps):
        if exper.args.problem == "quadratic":
            loss = val_set.compute_loss(average=False)
        else:
            loss = val_set.compute_neg_ll(average_over_funcs=False, size_average=False)
        if save_qt_prob_funcs:
            exper.val_stats["loss_funcs"][:, i] = loss.data.cpu().squeeze().numpy()
        if verbose and not exper.args.learner == 'manual' and i % 2 == 0:
            for f_idx in plot_idx:
                meta_logger.info("\tStep {}: current loss {:.4f}".format(str(i+1), loss.squeeze().data[f_idx].cpu()))

        loss.backward(backward_ones)
        param_loss = val_set.param_error(average=True).data.cpu().numpy()[0].astype(float)
        # remember dim 0 is batch size
        loss = torch.sum(torch.mean(loss, 0)).data.cpu().numpy()[0].astype(float)
        col_losses.append(loss)
        col_param_losses.append(param_loss)

        if not exper.args.learner == 'manual':

            delta_p = meta_learner.forward(val_set.params.grad.view(-1))
            # delta_p = meta_learner.meta_update(val_set)
            if exper.args.learner == 'meta':
                # gradient descent
                par_new = val_set.params - delta_p
                if exper.args.problem == "quadratic":
                    loss_step = val_set.compute_loss(average=True, params=par_new)
                    meta_learner.losses.append(Variable(loss_step.data))
                else:
                    # Regression
                    _ = meta_learner.step_loss(val_set, par_new, average_batch=True)

            elif exper.args.learner == 'act':
                # in this case forward returns a tuple (parm_delta, qt)
                param_size = val_set.params.size()
                par_new = val_set.params - delta_p[0].view(param_size)
                qt_delta = torch.mean(delta_p[1].view(param_size), 1)
                qt_param = qt_param + qt_delta
                qt_weights.append(qt_param.data.cpu().numpy().astype(float))
                # actually only calculating step loss here meta_leaner will collect the losses in order to
                # compute the final ACT loss
                if exper.args.problem == "quadratic":
                    loss_step = val_set.compute_loss(average=False, params=par_new)
                    val_set.losses.append(loss_step)

                else:
                    # Regression
                    loss_step = meta_learner.step_loss(val_set, par_new, average_batch=False)

                meta_learner.q_t.append(qt_param)
                # we're currently not breaking out of the loop when do_stop is true, therefore we
                # need this extra do_stop condition here in order not to compute it again
                if len(qt_weights) >= 2:
                    q_logits = np.concatenate(qt_weights, 1)
                    q_probs = softmax(np.array(q_logits))
                    do_stop = stop_computing(q_probs, threshold=exper.config.qt_threshold)
                    # register q(t|T) statistics
                    meta_learner.qt_hist_val[i+1] += np.mean(q_probs, 0)
                    meta_learner.opt_step_hist_val[i+1 - 1] += 1
                    if save_qt_prob_funcs:
                        exper.val_stats["qt_funcs"][i+1] = q_probs

            # Update the parameter of the function that is optimized
            val_set.set_parameters(par_new)

        else:
            meta_learner.step()

        # collected the loss and parameter-error

        val_set.params.grad.data.zero_()
        # increase the opt steps variable per function in case do_stop entry is False
        # NOTE: otherwise opt_steps will NOT BE INCREASED which means the value registered for this particular
        # function is equal to the stopping step (current t - 1) which should be correct because we tested
        # np.cumsum(qt_probs)[:, -2] > threshold, so is the qt-value t-1 bigger than our preset threshold
        opt_steps += np.logical_not(do_stop)
        """
            ************ END OF A VALIDATION OPTIMIZATION  ******************

        """

    # make another step to register final loss
    if exper.args.problem == "quadratic":
        loss = val_set.compute_loss(average=False)
    else:
        loss = val_set.compute_neg_ll(average_over_funcs=False, size_average=False)

    if save_qt_prob_funcs:
        exper.val_stats["loss_funcs"][:, i+1] = loss.data.cpu().squeeze().numpy()

    diff_min = torch.mean(loss - val_set.true_minimum_nll.expand_as(loss)).data.cpu().squeeze().numpy()[0].astype(float)
    loss = torch.sum(torch.mean(loss, 0)).data.cpu().squeeze().numpy()[0].astype(float)
    param_loss = val_set.param_error(average=True).data.cpu().numpy().astype(float)
    # add to total loss
    col_losses.append(loss)
    col_param_losses.append(param_loss)

    np_losses = np.array(col_losses)
    np_param_losses = np.array(col_param_losses)
    total_loss = np.sum(np_losses)
    # concatenate losses into a string for plotting the gradient path
    if "step_losses" in exper.val_stats.keys():
        exper.val_stats["step_losses"][exper.epoch] += np_losses
    if "step_param_losses" in exper.val_stats.keys():
        exper.val_stats["step_param_losses"][exper.epoch] += np_param_losses

    str_losses = np.array_str(np_losses, precision=3, suppress_small=True)
    if exper.args.learner == "act":
        # get prior probs for this number of optimizer steps
        # TODO currently we set T=max_steps because we are not stopping at the optimal step!!!
        # TODO again max_steps need to be adjusted later here when we really stop!!!
        priors = construct_prior_p_t_T(max_steps, exper.config.ptT_shape_param, val_set.num_of_funcs, exper.args.cuda)
        total_act_loss = meta_learner.final_loss(prior_probs=priors, run_type='val').data.squeeze()[0]
        str_q_probs = np.array_str(np.around(softmax(np.array(qt_weights)), decimals=5))

    if verbose:
        for f in plot_idx:
            meta_logger.info("\tf{}: true parameter values: {})".format(str(i+1),
                                                                        np.array_str(val_set.true_params.data.numpy()[f, :])))
            meta_logger.info("\tf{}: final parameter values ({})".format(str(i+1),
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
    """
        ****************** END OF VALIDATION **********************
                Register results for this validation run
    """

    exper.val_stats["step_param_losses"][exper.epoch] = np.around(exper.val_stats["step_param_losses"][exper.epoch],
                                                                  decimals=3)
    exper.val_stats["step_losses"][exper.epoch] = np.around(exper.val_stats["step_losses"][exper.epoch],
                                                                  decimals=3)
    exper.val_stats["loss"].append(total_loss)

    end_validate = time.time()
    exper.val_stats["param_error"].append(param_loss)
    meta_logger.info("INFO - Epoch {}, elapsed time {:.2f} seconds: ".format(exper.epoch,
                                                                             (end_validate - start_validate)))
    meta_logger.info("INFO - Epoch {}: Final validation stats: total-step-losses / final-step loss / "
                     "final-true_min: {:.4}/{:.4}/{:.4}".format(exper.epoch, total_loss, loss, diff_min))
    if exper.args.learner == "act":
        exper.val_stats["ll_loss"][exper.epoch] = meta_learner.ll_loss
        exper.val_stats["kl_div"][exper.epoch] = meta_learner.kl_div
        exper.val_stats["kl_entropy"][exper.epoch] = meta_learner.kl_entropy
        exper.val_stats["act_loss"].append(total_act_loss)
        exper.val_avg_num_opt_steps = int(np.mean(opt_steps))
        meta_logger.info("INFO - Epoch {}: Final validation average ACT-loss: {:.4}".format(exper.epoch,
                                                                                            total_act_loss))
        meta_logger.info("INFO - Epoch {}: Average stopping-step: {}".format(exper.epoch, exper.val_avg_num_opt_steps))
        meta_logger.debug(
            "{} Epoch/Validation: CDF q(t) {}".format(exper.epoch, np.array_str(np.cumsum(np.mean(q_probs, 0)),
                                                                                precision=4)))
    # meta_logger.info("INFO - Epoch {}: Final step param-losses: {}".format(exper.epoch,
    #                 np.array_str(exper.val_stats["step_param_losses"][exper.epoch], precision=4)))
    meta_logger.info("INFO - Epoch {}: Final step losses: {}".format(exper.epoch,
                                                                           np.array_str(
                                                                               exper.val_stats["step_losses"][
                                                                                   exper.epoch], precision=4)))

    meta_logger.info("--------------------------- End of validation --------------------------------------------")
    if exper.args.learner != "manual" and save_model:
        model_path = os.path.join(exper.output_dir, meta_learner.name + "_vrun" + str(exper.epoch) +
                                  exper.config.save_ext)
        torch.save(meta_learner.state_dict(), model_path)
        meta_logger.info("INFO - Successfully saved model parameters to {}".format(model_path))
    if exper.args.learner == 'act':
        # save the results of the validation statistics
        exper.val_stats['qt_hist'][exper.epoch] = meta_learner.qt_hist_val
        exper.val_stats['opt_step_hist'][exper.epoch] = meta_learner.opt_step_hist_val
        # reset some variables of the meta learner otherwise the training procedures will have serious problems
        meta_learner.reset_final_loss()
    elif exper.args.learner == 'meta':
        meta_learner.reset_losses()

    if save_run is not None:
        save_exper(exper, file_name="exp_statistics_" + save_run + ".dll")

