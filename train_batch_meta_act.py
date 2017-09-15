import torch
from torch.autograd import Variable
import torch.nn as nn

from utils.common import OPTIMIZER_DICT, construct_prior_p_t_T, get_func_loss

STD_OPT_LR = 4e-1


def execute_batch(exper, optimizees, meta_optimizer, optimizer, epoch_obj, final_batch=False):

    func_is_nn_module = nn.Module in optimizees.__class__.__bases__
    # if we're using a standard optimizer
    if exper.args.learner == 'manual':
        meta_optimizer = OPTIMIZER_DICT[exper.args.optimizer]([optimizees.params], lr=STD_OPT_LR)

    # counter that we keep in order to enable BPTT
    forward_steps = 0
    sum_grads = 0
    num_of_backwards = 0
    # determine the number of optimization steps for this batch
    if exper.args.learner == 'meta' and exper.args.version[0:2] == 'V2':
        optimizer_steps = exper.pt_dist.rvs(n=1)[0]
    elif exper.args.learner == 'meta' and exper.args.version[0:2] == 'V7':
        # Curriculum learning
        optimizer_steps = exper.inc_learning_schedule[exper.epoch-1]

    elif exper.args.learner == 'act' and not exper.args.fixed_horizon:
        # sample T - the number of timesteps - from our PMF (note prob to continue is set in config object)
        # add one to choice because we actually want values between [1, config.T]
        optimizer_steps = exper.pt_dist.rvs(n=1)[0]
        epoch_obj.prior_probs = construct_prior_p_t_T(optimizer_steps, exper.config.ptT_shape_param,
                                                      exper.args.batch_size,
                                                      exper.args.cuda)
    else:
        optimizer_steps = exper.args.optimizer_steps
        epoch_obj.avg_opt_steps = [exper.args.optimizer_steps]

    epoch_obj.avg_opt_steps.append(optimizer_steps)

    # the q-parameter for the ACT model, initialize
    qt_param = Variable(torch.zeros(exper.args.batch_size, 1))
    if exper.args.cuda:
        qt_param = qt_param.cuda()

    # outer loop with the optimization steps
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
                optimizees.reset_params()
                loss_sum = 0
            else:
                forward_steps += 1
        elif exper.args.learner == 'act' and k == 0:
            # ACT model: the LSTM hidden states will be only initialized
            # for the first optimization step
            forward_steps = 1
            # initialize LSTM
            meta_optimizer.reset_lstm(keep_states=False)
            optimizees.reset_params()
            loss_sum = 0
        # compute loss and generate gradients of optimizee
        loss = get_func_loss(exper, optimizees, average=False)
        if exper.args.problem == "mlp":
            loss.backward()
            # print("Loss {} & Grads {}".format(loss.data[0], torch.sum(optimizees.get_flat_grads()).data[0]))
            avg_loss = loss.data.cpu().squeeze().numpy()[0].astype(float)
        else:
            # compute gradients of optimizee which will need for the meta-learner
            loss.backward(epoch_obj.backward_ones)
            avg_loss = torch.mean(loss, 0).data.cpu().squeeze().numpy()[0].astype(float)

        exper.epoch_stats["step_losses"][exper.epoch][k] += avg_loss
        exper.epoch_stats["opt_step_hist"][exper.epoch][k] += 1
        epoch_obj.total_loss_steps += avg_loss
        # V6 improvement
        if exper.args.learner == "meta" and k == 0 and exper.args.version == "V6":
            loss_sum = Variable(torch.mean(loss.data.squeeze(), 0))
            # we only need to append here for t_0 because the .step_loss function adds t_i's
            meta_optimizer.losses.append(Variable(torch.mean(loss, 0).data.squeeze()))

        # meta_logger.info("{}/{} Sum optimizee gradients {:.3f}".format(
        #    i, k, torch.sum(optimizees.params.grad.data)))
        # feed the RNN with the gradient of the error surface function
        if exper.args.learner == 'meta':
            delta_param = meta_optimizer.meta_update(optimizees)
            if exper.args.problem == "quadratic":
                par_new = optimizees.params - delta_param
                loss_step = optimizees.compute_loss(average=True, params=par_new)
                meta_optimizer.losses.append(Variable(loss_step.data))
            elif exper.args.problem[0:10] == "regression":
                # Regression
                par_new = optimizees.params - delta_param
                loss_step = meta_optimizer.step_loss(optimizees, par_new, average_batch=True)

                if exper.args.learner == "meta" and exper.args.version == "V6":
                    # V6 observed improvement
                    meta_optimizer.losses[-1] = loss_step
                    min_f = torch.min(torch.cat(meta_optimizer.losses[0:k + 1], 0))
                    observed_imp = loss_step - min_f
                else:
                    observed_imp = None

            elif exper.args.problem == "rosenbrock":
                if exper.args.version[0:2] == "V4":
                    # metaV4, meta_update returns tuple (delta_param, qt-value)
                    par_new = optimizees.get_flat_params() + delta_param[0]
                    loss_step = torch.mean(delta_param[1] * optimizees.evaluate(parameters=par_new,
                                                                                average=False), 0).squeeze()
                else:
                    par_new = optimizees.get_flat_params() + delta_param
                    loss_step = optimizees.evaluate(parameters=par_new, average=True)
                meta_optimizer.losses.append(Variable(loss_step.data.unsqueeze(1)))

            elif exper.args.problem == "mlp":
                par_new = optimizees.get_flat_params() + delta_param.unsqueeze(1)
                optimizees.set_eval_obj_parameters(par_new)
                image, y_true = exper.dta_set.next_batch(is_train=True)
                loss_step = optimizees.evaluate(image , use_copy_obj=True, compute_loss=True, y_true=y_true)
                meta_optimizer.losses.append(Variable(loss_step.data.unsqueeze(1)))

            optimizees.set_parameters(par_new)
            if exper.args.version[0:2] == "V3":
                loss_sum = loss_sum + torch.mul(exper.fixed_weights[k], loss_step)

            elif exper.args.learner == "meta" and exper.args.version == "V6":
                loss_sum = loss_sum + observed_imp
            else:
                loss_sum = loss_sum + loss_step
        # ACT model processing. NOTE: we only end up here if learner != "meta" !!!
        elif exper.args.learner == 'act':
            delta_param, delta_qt = meta_optimizer.meta_update(optimizees)

            if exper.args.problem == "mlp":
                par_new = optimizees.get_flat_params() + delta_param.unsqueeze(1)
                delta_qt = torch.mean(delta_qt, 0, keepdim=True)
                optimizees.set_eval_obj_parameters(par_new)
                image, y_true = exper.dta_set.next_batch(is_train=True)
                loss_step = optimizees.evaluate(image , use_copy_obj=True, compute_loss=True, y_true=y_true)
                meta_optimizer.losses.append(Variable(loss_step.data.unsqueeze(1)))
            else:
                par_new = optimizees.params - delta_param

            qt_param = qt_param + delta_qt
            if exper.args.problem == "quadratic":
                loss_step = optimizees.compute_loss(average=False, params=par_new)
                meta_optimizer.losses.append(loss_step)
                loss_step = 1 / float(optimizees.num_of_funcs) * torch.sum(loss_step)
            elif exper.args.problem != "mlp":
                # Regression
                loss_step = meta_optimizer.step_loss(optimizees, par_new, average_batch=True)
            meta_optimizer.q_t.append(qt_param)
            loss_sum = loss_sum + loss_step
            optimizees.set_parameters(par_new)
        else:
            # NOTE: we only end up here if learer != "meta" and != "act"
            # we're just using one of the pre-delivered optimizers, update function parameters
            meta_optimizer.step()
            # compute loss after update
            loss_step = optimizees.compute_neg_ll(average_over_funcs=False, size_average=False)

        # set gradients of optimizee to zero again
        if func_is_nn_module:
            optimizees.zero_grad()
        else:
            optimizees.params.grad.data.zero_()
        # Check whether we need to execute BPTT
        if forward_steps == exper.args.truncated_bptt_step or k == optimizer_steps - 1:
            # meta_logger.info("BPTT at {}".format(k + 1))
            if exper.args.learner == 'meta' or (exper.args.learner == 'act'
                                                and exper.args.version[0:2] == "V1"):
                # meta_logger.info("{}/{} Sum loss {:.3f}".format(i, k,
                # loss_sum.data.cpu().squeeze().numpy()[0]))
                if exper.args.learner == 'meta' and exper.args.version[0:2] == "V5":
                    # in this version we make sure we never execute BPTT, we calculate the cumulative
                    # discounted reward at time step T (backward sweep)
                    loss_sum = meta_optimizer.final_loss(exper.fixed_weights)

                loss_sum.backward()
                num_of_backwards += 1
                optimizer.step()
                # save gradients if this is the last step, otherwise add to sum gradients
                if k == optimizer_steps - 1:
                    sum_grads += meta_optimizer.sum_grads()
                    epoch_obj.model_grads.append(sum_grads * 1./float(num_of_backwards))
                else:
                    sum_grads += meta_optimizer.sum_grads()
                meta_optimizer.zero_grad()
                # Slightly sloppy. Actually for the ACTV1 model we only register the ACT loss as the
                # so called optimizer-loss. But actually ACTV1 is using both losses
                if exper.args.learner == 'meta':
                    epoch_obj.loss_optimizer += loss_sum.data.cpu().squeeze().numpy()[0]

    # END of iterative function optimization. Compute final losses and probabilities
    # compute the final loss error for this function between last loss calculated and function min-value
    error = loss_step.data.cpu().squeeze().numpy()[0]
    if hasattr(optimizees, "true_minimum_nll"):
        epoch_obj.diff_min += (loss_step -
                               optimizees.true_minimum_nll.expand_as(loss_step)).data.cpu().squeeze().numpy()[0].astype(
            float)
    avg_loss = loss_step.data.cpu().squeeze().numpy()[0]
    exper.epoch_stats["step_losses"][exper.epoch][k + 1] += avg_loss
    exper.epoch_stats["opt_step_hist"][exper.epoch][k + 1] += 1
    epoch_obj.total_loss_steps += avg_loss
    # back-propagate ACT loss that was accumulated during optimization steps
    if exper.args.learner == 'act':
        # processing ACT loss
        act_loss = meta_optimizer.final_loss(epoch_obj.prior_probs, run_type='train')
        act_loss.backward()
        optimizer.step()
        epoch_obj.final_act_loss += act_loss.data.cpu().squeeze().numpy()[0]
        # set grads of meta_optimizer to zero after update parameters
        epoch_obj.model_grads.append(meta_optimizer.sum_grads())
        meta_optimizer.zero_grad()
        epoch_obj.loss_optimizer += act_loss.data.cpu().squeeze().numpy()[0]

    if exper.args.learner == "act":
        meta_optimizer.reset_final_loss()
    elif exper.args.learner == 'meta':
        meta_optimizer.reset_losses()
    # END OF BATCH: FUNCTION OPTIMIZATION
    epoch_obj.loss_last_time_step += error
    if hasattr(optimizees, "param_error"):
        epoch_obj.param_loss += optimizees.param_error(average=True).data.cpu().squeeze().numpy()[0]

    if exper.args.problem == "mlp" and final_batch:
        # evaluate the last MLP that we optimized
        accuracy = optimizees.test_model(exper.dta_set, exper.args.cuda, quick_test=True)
        exper.meta_logger.info("Note: End of batch - Accuracy of last MLP {:.4f}".format(accuracy))


