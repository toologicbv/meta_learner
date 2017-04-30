from config import config
import matplotlib.pyplot as plt

import os
import numpy as np
from collections import OrderedDict
from datetime import datetime
from pytz import timezone
from itertools import cycle
from probs import ConditionalTimeStepDist
from torch.autograd import Variable


def create_exper_label(exper):

    retrain = "_retrain" if exper.args.retrain else ""
    label1 = exper.args.learner + exper.args.version + "_" + str(exper.args.max_epoch) + "ep_" + \
        str(int(exper.avg_num_opt_steps)) + "ops" + retrain

    return label1


def create_x_val_array(exper, loss):
    x_vals = [i for i in range(exper.epoch + 1) if i % exper.args.eval_freq == 0]

    if exper.args.eval_freq == 1:
        x_vals = x_vals[1:]
    elif exper.args.eval_freq == exper.epoch:
        # eval frequency and total number of epochs are the same, therefore
        # one evaluation moments exper.epoch
        x_vals = [exper.epoch]
    else:
        x_vals = x_vals[1:]
        if len(x_vals) == len(loss):
            if x_vals[-1] != exper.epoch:
                x_vals[-1] = exper.epoch
        else:
            if x_vals[-1] != exper.epoch:
                x_vals.append(exper.epoch)
    return x_vals


def get_exper_loss_data(exper, loss_type, fig_name=None):
    exper_label = "_" + create_exper_label(exper)
    dt = "_" + datetime.now(timezone('Europe/Berlin')).strftime('%Y-%m-%d %H:%M:%S.%f')[-15:-7]
    if loss_type == "loss":
        if exper.args.learner == "act":
            num_opt_steps = exper.avg_num_opt_steps
        else:
            num_opt_steps = exper.args.optimizer_steps

        train_loss = exper.epoch_stats['loss']
        val_loss = exper.val_stats['loss']
        plt.ylabel("loss")
        if fig_name is None:
            fig_name = os.path.join(exper.output_dir, config.loss_fig_name + exper_label + dt + config.dflt_plot_ext)
    elif loss_type == "act_loss":
        num_opt_steps = exper.avg_num_opt_steps
        train_loss = exper.epoch_stats['act_loss']
        val_loss = exper.val_stats['act_loss']
        plt.ylabel("act-loss")
        if fig_name is None:
            fig_name = os.path.join(exper.output_dir, config.act_loss_fig_name + exper_label + dt +
                                    config.dflt_plot_ext)
    elif loss_type == "param_error":
        num_opt_steps = exper.avg_num_opt_steps
        train_loss = exper.epoch_stats['param_error']
        val_loss = exper.val_stats['param_error']
        plt.ylabel("Parameter error")
        if fig_name is None:
            fig_name = os.path.join(exper.output_dir, config.param_error_fig_name + exper_label + dt + config.dflt_plot_ext)
    else:
        raise ValueError("loss_type {} not supported".format(loss_type))

    return num_opt_steps, train_loss, val_loss, fig_name


def loss_plot(exper, fig_name=None, loss_type="loss", height=8, width=6, save=False, show=False, validation=True):

    num_opt_steps, train_loss, val_loss, fig_name = get_exper_loss_data(exper, loss_type, fig_name=fig_name)
    title_font = {'fontname': 'Arial', 'size': '14', 'color': 'black', 'weight': 'normal'}
    plt.figure(figsize=(height, width))
    plt.xlabel("epochs")
    x_vals = range(1, exper.epoch+1, 1)
    plt.plot(x_vals[2:], train_loss[2:], 'r', label="train-loss")

    if len(x_vals) <= 10:
        plt.xticks(x_vals)

    x_vals = create_x_val_array(exper, val_loss)

    if x_vals[0] <= 5:
        offset = 1
    else:
        offset = 0

    if validation:
        plt.plot(x_vals[offset:], val_loss[offset:], 'b', label="valid-loss")
    plt.legend(loc="best")
    plt.title("Train/validation loss {}-epochs/{}-opt-steps".format(exper.epoch, num_opt_steps), **title_font)
    if save:
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if show:
        plt.show()
    plt.close()


def param_error_plot(exper, fig_name=None, height=8, width=6, save=False, show=False):

    exper_label = "_" + create_exper_label(exper)
    dt = "_" + datetime.now(timezone('Europe/Berlin')).strftime('%Y-%m-%d %H:%M:%S.%f')[-15:-7]
    title_font = {'fontname': 'Arial', 'size': '14', 'color': 'black', 'weight': 'normal'}
    plt.figure(figsize=(height, width))
    plt.xlabel("epochs")
    plt.ylabel("parameter error")
    x_vals = range(1, exper.epoch+1, 1)
    plt.plot(x_vals[2:], exper.epoch_stats['param_error'][2:], 'r', label="train-loss")

    if len(x_vals) <= 10:
        plt.xticks(x_vals)

    x_vals = create_x_val_array(exper, exper.val_stats['param_error'])
    if x_vals[0] <= 5:
        offset = 1
    else:
        offset = 0
    plt.plot(x_vals[offset:], exper.val_stats['param_error'][offset:], 'b', label="valid-loss")
    plt.legend(loc="best")
    plt.title("Train/validation param-error {}-epochs/{}-opt-steps".format(exper.epoch, exper.avg_num_opt_steps),
              **title_font)
    if fig_name is None:
        fig_name = os.path.join(exper.output_dir, config.param_error_fig_name + exper_label + dt + config.dflt_plot_ext)
    if save:
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if show:
        plt.show()
    plt.close()


def plot_dist_optimization_steps(exper, data_set="train", fig_name=None, height=8, width=6, save=False, show=False):
    title_font = {'fontname': 'Arial', 'size': '14', 'color': 'black', 'weight': 'normal'}
    bar_width = 0.5
    if data_set == "train":
        opt_step_hist = exper.epoch_stats["opt_step_hist"]
    else:
        opt_step_hist = exper.val_stats["opt_step_hist"]
    # because we shift the distribution by 1 to start with t=1 until config.T we also need to increase the
    # indices here
    index = range(1, len(opt_step_hist) + 1)
    norms = 1. / np.sum(opt_step_hist) * opt_step_hist
    o_mean = int(round(np.sum(index * norms)))
    plt.figure(figsize=(height, width))
    plt.bar(index, norms, bar_width, color='b', align='center',
            label="with q(continue)={:.3f}".format(config.continue_prob))
    # plot mean value again...in red
    plt.bar([o_mean], norms[o_mean - 1], bar_width, color='r', align="center")
    plt.xlabel("Number of optimization steps")
    plt.ylabel("Proportion")
    plt.title("Distribution of optimization steps (E[T|{}]={})".format(config.T, o_mean), **title_font)
    plt.legend(loc="best")
    if fig_name is None:
        fig_name = os.path.join(exper.output_dir, config.T_dist_fig_name + "_" + data_set + config.dflt_plot_ext)
    if save:
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if show:
        plt.show()
    plt.close()


def plot_qt_probs(exper, data_set="train", fig_name=None, height=16, width=12, save=False, show=False, plot_idx=[],
                  plot_prior=False):
    bar_width = 0.3
    if data_set == "train":
        T = len(exper.epoch_stats["qt_hist"])
        opt_step_hist = exper.epoch_stats["opt_step_hist"]
        qt_hist = exper.epoch_stats["qt_hist"]
        plot_title = "Training - q(t) distributions for different T (mean={})".format(int(exper.avg_num_opt_steps))
    else:
        T = len(exper.val_stats["qt_hist"])
        opt_step_hist = exper.val_stats["opt_step_hist"]
        qt_hist = exper.val_stats["qt_hist"]
        plot_title = "Validation - q(t) distributions "

    res_qts = OrderedDict()
    for i in range(1, T + 1):
        if opt_step_hist[i - 1] != 0:
            res_qts[i] = qt_hist[i] * 1. / opt_step_hist[i - 1]
        else:
            res_qts[i] = []

    if len(plot_idx) == 0:
        plot_idx = [int(exper.avg_num_opt_steps + i) for i in range(-config.qt_mean_range, config.qt_mean_range+1)]
        # in case idx=1 in list remove, because probs for 1 step are always 1.
        if 1 in plot_idx:
            plot_idx.remove(1)

        if data_set == 'val':
            # for now we make only one plot for the validation statistics because we fixed the # of steps to 4
            # determine number of plots we can make
            if np.sum(opt_step_hist != 0) == 1:
                num_of_plots = 1
                height = 10
                width = 25
                plot_idx = [np.argmax(opt_step_hist) + 1]
            elif np.sum(opt_step_hist > 5) == 1:
                plot_idx = [np.argmax(opt_step_hist) + 1]
                num_of_plots = len(plot_idx)
            else:
                if np.sum(opt_step_hist > 5) > 3:
                    num_of_plots = 3
                else:
                    num_of_plots = int(np.sum(opt_step_hist > 5))
                plot_idx = list(np.argsort(opt_step_hist)[::-1][0:num_of_plots] + 1)

        else:
            # training, always enough values at our hand
            check = plot_idx[:]
            for idx in check:
                # if we don't have any results for this number of steps than remove from list
                if idx not in res_qts:
                    plot_idx.remove(idx)

            num_of_plots = len(plot_idx)
    else:
        num_of_plots = len(plot_idx)

    fig = plt.figure(figsize=(width, height))
    fig.suptitle(plot_title, **config.title_font)
    for i in range(1, num_of_plots + 1):
        index = range(1, plot_idx[i - 1] + 1)
        T = len(index)
        if i == 1:
            ax1 = plt.subplot(num_of_plots, 2, i)
        else:
            _ = plt.subplot(num_of_plots, 2, i, sharey=ax1)

        plt.bar(index, res_qts[plot_idx[i - 1]], bar_width, color='b', align='center')
        if plot_prior:
            kl_prior_dist = ConditionalTimeStepDist(T=T, q_prob=config.continue_prob)
            priors = kl_prior_dist.pmfunc(np.arange(0, T))
            plt.bar(np.array(index)+bar_width, priors, bar_width, color='r', align='center')
        plt.xticks(index)
        if i == num_of_plots or i == num_of_plots - 1:
            plt.xlabel("Steps")
        if i % 2 == 0:
            # TODO hide yticks for this subplots, looks better
            pass
        else:
            plt.ylabel("Probs")

    if fig_name is None:
        fig_name = "_" + data_set + "_" + create_exper_label(exper)
        fig_name = os.path.join(exper.output_dir, config.qt_dist_prefix + fig_name + ".png")
    if save:
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if show:
        plt.show()
    plt.close()


def plot_val_result(expers, height=8, width=12, do_show=True, do_save=False, fig_name=None, plot_best=False,
                    loss_type='param_error'):
    num_of_expers = len(expers)
    title_font = {'fontname': 'Arial', 'size': '14', 'color': 'black', 'weight': 'normal'}

    best_val_runs = [0 for i in range(num_of_expers)]
    lowest_value = [0 for i in range(num_of_expers)]
    idx_lowest_value = [0 for i in range(num_of_expers)]
    p_colors = ['grey', 'orange', 'red', 'violet', 'dodgerblue', 'green', 'darkviolet']

    plt.figure(figsize=(width, height))
    style = [[8, 4, 2, 4, 2, 4], [4, 2, 2, 4, 2, 4], [2, 2, 2, 4, 2, 4], [4, 8, 2, 4, 2, 4], [8, 8, 8, 4, 2, 4],
             [4, 4, 2, 8, 2, 2]]
    iter_colors = cycle(p_colors)
    iter_styles = cycle(style)

    for e in np.arange(num_of_expers):
        if loss_type == "param_error":
            res_dict = expers[e].val_stats["step_param_losses"]
            y_label = "avg final param error"
            plot_title = "Final avg parameter error per step for 2D-Quadratics"
        else:
            res_dict = expers[e].val_stats["step_losses"]
            y_label = "avg final loss"
            plot_title = "Final avg loss per step for 2D-Quadratics"
        if plot_best:
            plot_title = "Best runs selected --- " + plot_title
        else:
            plot_title = "Last run selected --- " + plot_title

        keys = res_dict.keys()
        # res = res_dict[keys[len(keys) - i]]
        model = expers[e].args.model
        min_param_value = 999.

        if plot_best:
            for val_run in keys:
                idx_min = np.argmin(res_dict[val_run])
                if min_param_value > res_dict[val_run][idx_min]:
                    best_val_runs[e] = val_run
                    lowest_value[e] = res_dict[val_run][idx_min]
                    min_param_value = lowest_value[e]
                    idx_lowest_value[e] = idx_min
            print("model: {} best val run {} = {:.4f} / step {}".format(model, best_val_runs[e],
                                                                        lowest_value[e], idx_lowest_value[e]))
        else:
            # get the last validation runs, we're just plotting the last run...not the best
            val_run = keys[-1]
            best_val_runs[e] = val_run
            lowest_value[e] = res_dict[val_run][-1]

        if hasattr(expers[e], "val_avg_num_opt_steps"):
            stop_step = expers[e].val_avg_num_opt_steps
            """
                TO DO, plot the step where ACT would stop on average
            """
            # print("val_avg_num_opt_steps {}".format(expers[e].val_avg_num_opt_steps))
        else:
            stop_step = config.max_val_opt_steps
        index = np.arange(1, len(res_dict[best_val_runs[e]]) + 1)

        #

        plt.semilogy(index, res_dict[best_val_runs[e]], color=iter_colors.next(), dashes=iter_styles.next(),
                     linewidth=2., label="{}({})(stop={})".format(model, best_val_runs[e], stop_step))
        plt.xticks(index, index - 1)
        plt.xlabel("Number of optimization steps")
        plt.ylabel(y_label)
        plt.legend(loc="best")
        plt.title(plot_title, **title_font)

    if do_save:
        dt = "_" + datetime.now(timezone('Europe/Berlin')).strftime('%Y-%m-%d %H:%M:%S.%f')[-15:-7]
        dt = str.replace(str.replace(dt, ':', '_'), '-', '')
        if fig_name is None:
            fig_name = "best_val_results"
        fig_name = os.path.join(config.figure_path, fig_name + dt + config.dflt_plot_ext)

        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if do_show:
        plt.show()

    plt.close()

    return best_val_runs, lowest_value


def plot_exper_losses(expers, loss_type="loss", height=8, width=12, do_show=True, do_save=False, offset=5,
                      N=4, fig_name=None, validation=False):

    num_of_expers = len(expers)
    title_font = {'fontname': 'Arial', 'size': '14', 'color': 'black', 'weight': 'normal'}

    p_colors = ['grey', 'orange', 'red', 'violet', 'dodgerblue', 'green', 'darkviolet']
    plot_title = "Avg " + "validation" if validation else "training"  
    plot_title += " loss per epoch for 2D-Quadratics"
    plt.figure(figsize=(width, height))
    plt.title(plot_title, **title_font)
    style = [[8, 4, 2, 4, 2, 4], [4, 2, 2, 4, 2, 4], [2, 2, 2, 4, 2, 4], [4, 8, 2, 4, 2, 4], [8, 8, 8, 4, 2, 4],
             [4, 4, 2, 8, 2, 2]]
    iter_colors = cycle(p_colors)

    for e in np.arange(num_of_expers):

        if loss_type == "loss":
            loss = "loss"
        elif loss_type == 'act_loss':
            loss = 'act_loss'
        elif loss_type == 'param_error':
            loss = "param_error"
        else:
            raise ValueError("{} loss type is not supported".format(loss_type))

        num_opt_steps, train_loss, val_loss, _ = get_exper_loss_data(expers[e], loss)

        model = expers[e].args.model

        if not validation:
            # training losses
            x_vals = range(1, expers[e].epoch + 1, 1)
            # act is only valid for act learners
            if train_loss == []:
                pass
            else:
                avg_train_loss = np.convolve(train_loss, np.ones((N,)) / N, mode='same')
                plt.plot(x_vals[offset:], avg_train_loss[offset:], color=iter_colors.next(), linewidth=2.,
                        label="{}({})".format(model, expers[e].epoch))
        else:
            # validation results
            x_vals = create_x_val_array(expers[e], val_loss)

            if x_vals[0] <= 5:
                offset = 1
            else:
                offset = 0
            if val_loss == []:
                pass
            else:
                plt.plot(x_vals[offset:], val_loss[offset:], color=iter_colors.next(), linewidth=2., 
                    label="{}({})".format(model, expers[e].epoch))

        if len(x_vals) <= 10:
            plt.xticks(x_vals)

        plt.xlabel("Epochs")
        # plt.ylabel("Loss")
        plt.legend(loc="best")

    if do_save:
        dt = "_" + datetime.now(timezone('Europe/Berlin')).strftime('%Y-%m-%d %H:%M:%S.%f')[-15:-7]
        dt = str.replace(str.replace(dt, ':', '_'), '-', '')
        if fig_name is None:
            fig_name = "loss_curves"
        fig_name = os.path.join(config.figure_path, fig_name + dt + config.dflt_plot_ext)

        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)

    if do_show:
        plt.show()
    plt.close()


def plot_parm_loss_steps(expers, height=25, width=15, num_of_plots=0, do_show=False, fig_name=None, do_save=False):
    num_of_expers = len(expers)
    num_of_val_runs = len(expers[0].val_stats["step_param_losses"])

    if num_of_plots == 0:
        num_of_plots = num_of_val_runs

    max_steps = len(expers[0].val_stats["step_param_losses"][expers[0].val_stats["step_param_losses"].keys()[0]])
    index = np.arange(1, max_steps + 1)

    plot_title = "Final avg parameter error for 2D-Quadratics"
    fig = plt.figure(figsize=(width, height))
    fig.suptitle(plot_title, **config.title_font)

    p_colors = ['grey', 'orange', 'red', 'violet', 'dodgerblue', 'green', 'darkviolet']
    style = [[8, 4, 2, 4, 2, 4], [4, 2, 2, 4, 2, 4], [2, 2, 2, 4, 2, 4], [4, 8, 2, 4, 2, 4], [8, 8, 8, 4, 2, 4],
             [4, 4, 2, 8, 2, 2]]

    if num_of_plots == 1:
        plot_cols = 1
        fig.set_figheight(9)
    elif num_of_plots == 2:
        plot_cols = 1
        fig.set_figheight(15)
    else:
        plot_cols = 2

    for i in np.arange(1, num_of_plots + 1):

        if i == 1:
            ax1 = plt.subplot(num_of_plots, plot_cols, i)
        else:
            _ = plt.subplot(num_of_plots, plot_cols, i, sharey=ax1)

        if i == num_of_plots or i == num_of_plots - 1:
            plt.xlabel("Number of optimization steps")
        if i % 2 != 0:
            plt.ylabel("Final error")
        if num_of_plots > 2:
            fig.set_figheight(25)
        iter_colors = cycle(p_colors)
        iter_styles = cycle(style)
        for e in np.arange(num_of_expers):
            res_dict = expers[e].val_stats["step_param_losses"]
            keys = res_dict.keys()
            # could be that not all experiments have the same number of val runs
            if i <= len(keys):
                res = res_dict[keys[len(keys) - i]]
                model = expers[e].args.model
                # plt.bar(index + (e*bar_width), res,  bar_width,  color=bar_colors[e], log=True,
                #        align="center", label="{}({})".format(model, keys[len(keys)-i]))
                # plt.xticks(index + bar_width / 2, index-1)
                plt.semilogy(index, res, color=iter_colors.next(), dashes=iter_styles.next(), linewidth=2.,
                             label="{}({})".format(model, keys[len(keys) - i]))
                plt.legend(loc="best", prop={'size': 8})

    if do_save:
        dt = "_" + datetime.now(timezone('Europe/Berlin')).strftime('%Y-%m-%d %H:%M:%S.%f')[-15:-7]
        dt = str.replace(str.replace(dt, ':', '_'), '-', '')
        if fig_name is None:
            fig_name = "param_loss_per_step"
        fig_name = os.path.join(config.figure_path, fig_name + dt + config.dflt_plot_ext)

        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)

    if do_show:
        plt.show()
    plt.close()
