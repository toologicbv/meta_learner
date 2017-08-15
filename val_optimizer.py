import os
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils.common import load_val_data

from utils.common import softmax, stop_computing, construct_prior_p_t_T, generate_fixed_weights, \
                        get_func_loss


def validate_optimizer(meta_learner, exper, meta_logger, val_set=None, max_steps=6, verbose=True, plot_func=False,
                       num_of_plots=3, save_plot=True, show_plot=False, save_qt_prob_funcs=False, save_model=False,
                       save_run=None):

    start_validate = time.time()
    global STD_OPT_LR
    # we will probably call this procedure later in another context (to evaluate meta-learners)
    # so make sure the globals exist.
    if 'STD_OPT_LR' not in globals():
        STD_OPT_LR = 4e-1
    # initialize stats arrays
    exper.val_stats["step_losses"][exper.epoch] = np.zeros(exper.config.max_val_opt_steps + 1)
    exper.val_stats["step_param_losses"][exper.epoch] = np.zeros(exper.config.max_val_opt_steps + 1)

    exper.meta_logger.info("---------------------------------------------------------------------------------------")
    if val_set is None:
        # if no validation set is provided just use one random generated q-function to run the validation
        exper.meta_logger.info("INFO - No validation set provided, generating new functions")
        val_set = load_val_data(num_of_funcs=exper.config.num_val_funcs, n_samples=exper.args.x_samples,
                                stddev=exper.config.stddev, dim=exper.args.x_dim, logger=exper.meta_logger,
                                exper=exper)

        plot_idx = [0]
    else:
        plot_idx = [(i + 1) * (val_set.num_of_funcs // num_of_plots) - 1 for i in range(num_of_plots)]

    exper.meta_logger.info("INFO - Epoch {}: Validating model {} with {} functions".format(exper.epoch, exper.args.model,
                                                                                     val_set.num_of_funcs))
    total_opt_loss = 0
    func_is_nn_module = nn.Module in val_set.__class__.__bases__
    # IMPORTANT - resetting hidden states and validation functions (set back to initial values)
    meta_learner.reset_lstm(keep_states=False)
    val_set.reset()
    if verbose:
        exper.meta_logger.info("\tStart-value parameters {}".format(np.array_str(val_set.params.data.numpy()[np.array(plot_idx)])))

    if exper.args.learner == 'manual':
        state_dict = meta_learner.state_dict()
        meta_learner = optim.Adam([val_set.params], lr=STD_OPT_LR)
        meta_learner.load_state_dict(state_dict)
    elif exper.args.learner == "act":
        meta_learner.reset_final_loss()
    elif exper.args.learner == "meta":
        meta_learner.reset_losses()
        fixed_weights = generate_fixed_weights(exper, steps=exper.config.max_val_opt_steps)

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
        loss = get_func_loss(exper, val_set, average=False)
        if save_qt_prob_funcs:
            exper.val_stats["loss_funcs"][:, i] = loss.data.cpu().squeeze().numpy()
        if verbose and not exper.args.learner == 'manual' and i % 2 == 0:
            for f_idx in plot_idx:
                exper.meta_logger.info("\tStep {}: current loss {:.4f}".format(str(i+1), loss.squeeze().data[f_idx].cpu()))

        loss.backward(backward_ones)
        param_loss = val_set.param_error(average=True).data.cpu().numpy()[0].astype(float)
        # remember dim 0 is batch size
        loss = torch.sum(torch.mean(loss, 0)).data.cpu().numpy()[0].astype(float)
        col_losses.append(loss)
        col_param_losses.append(param_loss)

        if not exper.args.learner == 'manual':
            if func_is_nn_module:
                # we already have flat parameters to pass to LSTM optimizer, no need for reshape
                grads = Variable(val_set.get_flat_grads().data)
                delta_p = meta_learner.forward(grads)
            else:
                # remaining part from meta-learner that uses grads+params as input to LSTM
                # delta_p = meta_learner.forward(torch.cat((val_set.params.grad.view(-1).unsqueeze(1), val_set.params.view(-1).unsqueeze(1)), 1))
                delta_p = meta_learner.forward(val_set.params.grad.view(-1))
            # delta_p = meta_learner.meta_update(val_set)
            if exper.args.learner == 'meta':
                if exper.args.problem == "quadratic":
                    # Quadratic function from L2L paper
                    param_size = val_set.params.size()
                    par_new = val_set.params - delta_p.view(param_size)
                    loss_step = val_set.compute_loss(average=True, params=par_new)
                    meta_learner.losses.append(Variable(loss_step.data))
                elif exper.args.problem[0:10] == "regression":
                    # Regression
                    param_size = val_set.params.size()
                    par_new = val_set.params - delta_p.view(param_size)
                    _ = meta_learner.step_loss(val_set, par_new, average_batch=True)
                elif exper.args.problem == "rosenbrock":
                    # Rosenbrock function
                    if exper.args.version[0:2] == "V4":
                        # metaV4, meta_update returns tuple (delta_param, qt-value)
                        par_new = val_set.get_flat_params() + delta_p[0]
                        param_size = list([val_set.num_of_funcs, val_set.dim])
                        delta_qt = torch.mean(delta_qt.view(*param_size), 1)
                        loss_step = torch.mean(delta_qt * val_set.evaluate(parameters=par_new,
                                                                             average=False), 0).squeeze()
                        meta_learner.q_t.append(delta_p[1].data.cpu().numpy())
                    else:
                        # not necessary to reshape the ouput of LSTM because we're working with flat parameter anyway
                        par_new = val_set.get_flat_params() + delta_p
                        loss_step = val_set.evaluate(parameters=par_new, average=False)
                    meta_learner.losses.append(Variable(loss_step.data.unsqueeze(1)))

            elif exper.args.learner == 'act':
                # in this case forward returns a tuple (parm_delta, qt)
                param_size = val_set.params.size()
                par_new = val_set.params - delta_p[0].view(param_size)
                qt_delta = torch.mean(delta_p[1].view(param_size), 1, keepdim=True)
                qt_param = qt_param + qt_delta
                qt_weights.append(qt_param.data.cpu().numpy().astype(float))
                # actually only calculating step loss here meta_leaner will collect the losses in order to
                # compute the final ACT loss
                if exper.args.problem == "quadratic":
                    loss_step = val_set.compute_loss(average=False, params=par_new)
                    val_set.losses.append(loss_step)
                else:
                    # Regression
                    _ = meta_learner.step_loss(val_set, par_new, average_batch=False)

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

        # set gradients of optimizee to zero again
        if func_is_nn_module:
            val_set.zero_grad()
        else:
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
    loss = get_func_loss(exper, val_set, average=False)

    if save_qt_prob_funcs:
        exper.val_stats["loss_funcs"][:, i+1] = loss.data.cpu().squeeze().numpy()
    # how many of the val funcs are close to global minimum? Temporary
    last_losses = loss.data.cpu().squeeze().numpy()
    # end
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
    exper.val_stats["step_losses"][exper.epoch] += np_losses
    exper.val_stats["step_param_losses"][exper.epoch] += np_param_losses

    str_losses = np.array_str(np_losses, precision=3, suppress_small=True)
    if exper.args.learner == "act":
        # get prior probs for this number of optimizer steps
        # TODO currently we set T=max_steps because we are not stopping at the optimal step!!!
        # TODO again max_steps need to be adjusted later here when we really stop!!!
        priors = construct_prior_p_t_T(max_steps, exper.config.ptT_shape_param, val_set.num_of_funcs, exper.args.cuda)
        total_opt_loss = meta_learner.final_loss(prior_probs=priors, run_type="val").data.squeeze()[0]
        str_q_probs = np.array_str(np.around(softmax(np.array(qt_weights)), decimals=5))
    elif exper.args.learner == "meta":
        total_opt_loss = meta_learner.final_loss(loss_weights=fixed_weights).data.squeeze()[0]

    if verbose:
        for f in plot_idx:
            exper.meta_logger.info("\tf{}: true parameter values: {})".format(str(i+1),
                                                                        np.array_str(val_set.true_params.data.numpy()[f, :])))
            exper.meta_logger.info("\tf{}: final parameter values ({})".format(str(i+1),
                                                                        np.array_str(val_set.params.data.numpy()[f, :])))
        if exper.args.learner == 'act':
            exper.meta_logger.info("Final qt-probabilities")
            exper.meta_logger.info("raw:   {}".format(np.array_str(np.array(qt_weights))))
            exper.meta_logger.info("probs: {}".format(str_q_probs))
            exper.meta_logger.info("losses: {}".format(str_losses))

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
    if "opt_loss" in exper.val_stats.keys():
        exper.val_stats["opt_loss"].append(total_opt_loss)
    duration = end_validate - start_validate
    exper.meta_logger.info("INFO - Epoch {}, elapsed time {:.2f} seconds: ".format(exper.epoch,
                                                                             duration))
    exper.meta_logger.info("INFO - Epoch {}: Final validation stats: total-step-losses / final-step loss / "
                     "final-true_min: {:.4}/{:.4}/{:.4}".format(exper.epoch, total_loss, loss, diff_min))
    exper.add_duration(duration, is_train=False)
    if exper.args.learner == "act":
        # exper.val_stats["ll_loss"][exper.epoch] = meta_learner.ll_loss
        # exper.val_stats["kl_div"][exper.epoch] = meta_learner.kl_div
        # exper.val_stats["kl_entropy"][exper.epoch] = meta_learner.kl_entropy
        exper.val_avg_num_opt_steps = int(np.mean(opt_steps))
        exper.meta_logger.info("INFO - Epoch {}: Final validation average ACT-loss: {:.4}".format(exper.epoch,
                                                                                            total_opt_loss))
        exper.meta_logger.info("INFO - Epoch {}: Average stopping-step: {}".format(exper.epoch, exper.val_avg_num_opt_steps))
        exper.meta_logger.debug(
            "{} Epoch/Validation: CDF q(t) {}".format(exper.epoch, np.array_str(np.cumsum(np.mean(q_probs, 0)),
                                                                                precision=4)))
    if exper.val_stats["step_losses"][exper.epoch].shape[0] > 210:
        step_results = exper.val_stats["step_losses"][exper.epoch][-100:]
        exper.meta_logger.info(">>> NOTE: only showing last 100 steps <<<")
    else:
        step_results = exper.val_stats["step_losses"][exper.epoch]
    exper.meta_logger.info("INFO - Epoch {}: Final step losses: {}".format(exper.epoch,
                                                                           np.array_str(step_results, precision=4)))

    exper.meta_logger.info("--------------------------- End of validation --------------------------------------------")
    if exper.args.learner != "manual" and save_model:
        model_path = os.path.join(exper.output_dir, meta_learner.name + "_vrun" + str(exper.epoch) +
                                  exper.config.save_ext)
        torch.save(meta_learner.state_dict(), model_path)
        exper.meta_logger.info("INFO - Successfully saved model parameters to {}".format(model_path))
    if exper.args.learner == 'act':
        # save the results of the validation statistics
        exper.val_stats['qt_hist'][exper.epoch] = meta_learner.qt_hist_val
        exper.val_stats['opt_step_hist'][exper.epoch] = meta_learner.opt_step_hist_val
        # reset some variables of the meta learner otherwise the training procedures will have serious problems
        meta_learner.reset_final_loss()
    elif exper.args.learner == 'meta':
        # Temporary
        # f_min_idx = np.where((last_losses >= -0.1) & (last_losses <= 0.1))
        if exper.args.problem == "rosenbrock":
            X, Y = val_set.split_parameters(val_set.initial_params)
            x_np = val_set.X.data.cpu().squeeze().numpy()
            y_np = val_set.Y.data.cpu().squeeze().numpy()
            a = val_set.a.data.cpu().squeeze().numpy()
            b = val_set.b.data.cpu().squeeze().numpy()
            a2 = a**2
            x_delta = np.abs(x_np - a)
            y_delta = np.abs(y_np - a**2)
            minx_idx = np.where((x_delta <= 0.1) & (x_delta >= 0.) & (a2 >= 0.1))
            miny_idx = np.where((y_delta <= 0.1) & (y_delta >= 0.) & (a2 >= 0.1))
            local_min = np.where((np.abs(x_np) <= 0.1) & (np.abs(y_np) <= 0.1))
            all = np.arange(val_set.num_of_funcs)
            x = set(minx_idx[0])
            y = set(miny_idx[0])
            r = list(x.intersection(y))
            r_comp = list(set(all) - set(r))
            exper.meta_logger.info("from {} close to global minimum {} - close to (0,0) {}".format(val_set.num_of_funcs,
                                                                                             len(r),
                                                                                             local_min[0].shape[0]))
            if len(r) > 0:
                try_idx = r[0]
                exper.meta_logger.info("function {} init({:.3f}, {:.3f}) "
                                 "true({:.3f}, {:.3f}) "
                                 "curr({:.3f}, {:.3f})".format(try_idx, X[try_idx].data.cpu().squeeze().numpy()[0],
                                                        Y[try_idx].data.cpu().squeeze().numpy()[0],
                                                        val_set.a[try_idx].data.cpu().squeeze().numpy()[0],
                                                        val_set.a[try_idx].data.cpu().squeeze().numpy()[0]**2,
                                                        val_set.X[try_idx].data.cpu().squeeze().numpy()[0],
                                                        val_set.Y[try_idx].data.cpu().squeeze().numpy()[0]))
            if False:
                parm_x = val_set.X.data.cpu().squeeze().numpy()
                parm_y = val_set.Y.data.cpu().squeeze().numpy()
                exper.meta_logger.info("true a")
                exper.meta_logger.info("{}".format(np.array_str(a[np.array(r_comp)[0:30]])))
                exper.meta_logger.info("true b")
                exper.meta_logger.info("{}".format(np.array_str(b[np.array(r_comp)[0:30]])))
                exper.meta_logger.info("current x")
                exper.meta_logger.info("{}".format(np.array_str(parm_x[np.array(r_comp)[0:30]])))
                exper.meta_logger.info("current y")
                exper.meta_logger.info("{}".format(np.array_str(parm_y[np.array(r_comp)[0:30]])))
                exper.meta_logger.info("current loss")
                exper.meta_logger.info("{}".format(np.array_str(last_losses[np.array(r_comp)[0:30]])))
                if exper.args.learner == "meta" and exper.args.version[0:2] == "V4":
                    qt = np.concatenate(meta_learner.q_t, 1)
                    exper.meta_logger.info("{}".format(np.array_str(qt[try_idx, :])))

        meta_learner.reset_losses()


