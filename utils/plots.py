from config import config
import matplotlib.pyplot as plt

import os
import numpy as np
from collections import OrderedDict
from datetime import datetime
from pytz import timezone


def create_exper_label(exper):

    retrain = "_retrain" if exper.args.retrain else ""
    label1 = exper.args.learner + exper.args.version + "_" + str(exper.args.max_epoch) + "ep_" + \
        str(int(exper.avg_num_opt_steps)) + "ops_" + exper.args.loss_type + retrain

    return label1


def loss_plot(exper, fig_name=None, loss_type="normal", height=8, width=6, save=False, show=False):
    title_font = {'fontname': 'Arial', 'size': '14', 'color': 'black', 'weight': 'normal'}
    plt.figure(figsize=(height, width))
    plt.xlabel("epochs")
    exper_label = "_" + create_exper_label(exper)
    dt = "_" + datetime.now(timezone('Europe/Berlin')).strftime('%Y-%m-%d %H:%M:%S.%f')[-15:-7]

    if loss_type == "normal":
        if exper.args.learner == "act":
            num_opt_steps = exper.avg_num_opt_steps
        else:
            num_opt_steps = exper.args.optimizer_steps

        train_loss = exper.epoch_stats['loss']
        val_loss = exper.val_stats['loss']
        plt.ylabel("loss")
        if fig_name is None:
            fig_name = os.path.join(exper.output_dir, config.loss_fig_name + exper_label + dt + config.dflt_plot_ext)
    elif loss_type == "act":
        num_opt_steps = exper.avg_num_opt_steps
        train_loss = exper.epoch_stats['act_loss']
        val_loss = exper.val_stats['act_loss']
        plt.ylabel("act-loss")
        if fig_name is None:
            fig_name = os.path.join(exper.output_dir, config.act_loss_fig_name + exper_label + dt +
                                    config.dflt_plot_ext)
    else:
        raise ValueError("loss_type {} not supported".format(loss_type))
    x_vals = range(1, exper.epoch+1, 1)
    plt.plot(x_vals, train_loss, 'r', label="train-loss")

    if len(x_vals) <= 10:
        plt.xticks(x_vals)
    x_vals = [i for i in range(exper.epoch+1) if i % exper.args.eval_freq == 0]
    if exper.args.eval_freq == 1:
        x_vals = x_vals[1:]
    elif exper.args.eval_freq == exper.epoch:
        # eval frequency and total number of epochs are the same, therefore
        # two evaluation moments 1 and exper.epoch
        x_vals = [1, exper.epoch]
    else:
        x_vals[0] = 1
        if len(x_vals) == len(val_loss):
            if x_vals[-1] != exper.epoch:
                x_vals[-1] = exper.epoch
        else:
            if x_vals[-1] != exper.epoch:
                x_vals.append(exper.epoch)

    plt.plot(x_vals, val_loss, 'b', label="valid-loss")
    plt.legend(loc="best")
    plt.title("Train/validation loss {}-epochs/{}-opt-steps/{}-loss-func".format(exper.epoch,
                                                                                 num_opt_steps,
                                                                                 exper.args.loss_type), **title_font)
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
    plt.plot(x_vals, exper.epoch_stats['param_error'], 'r', label="train-loss")

    if len(x_vals) <= 10:
        plt.xticks(x_vals)
    x_vals = [i for i in range(exper.epoch + 1) if i % exper.args.eval_freq == 0]
    if exper.args.eval_freq == 1:
        x_vals = x_vals[1:]
    elif exper.args.eval_freq == exper.epoch:
        # eval frequency and total number of epochs are the same, therefore
        # two evaluation moments 1 and exper.epoch
        x_vals = [1, exper.epoch]
    else:
        x_vals[0] = 1
        if len(x_vals) == len(exper.val_stats['param_error']):
            if x_vals[-1] != exper.epoch:
                x_vals[-1] = exper.epoch
        else:
            if x_vals[-1] != exper.epoch:
                x_vals.append(exper.epoch)

    plt.plot(x_vals, exper.val_stats['param_error'], 'b', label="valid-loss")
    plt.legend(loc="best")
    plt.title("Train/validation param-error {}-epochs/{}-opt-steps/{}-loss-func".format(exper.epoch,
                                                                                 exper.avg_num_opt_steps,
                                                                                 exper.args.loss_type), **title_font)
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
            label="p(T) distribution (q={:.3f})".format(config.continue_prob))
    # plot mean value again...in red
    plt.bar([o_mean], norms[o_mean - 1], bar_width, color='r', align="center")
    plt.xlabel("optimization steps")
    plt.ylabel("probability")
    plt.title("Distribution of optimization steps T (E[T|{}]={})".format(config.T, o_mean), **title_font)
    plt.legend(loc="best")
    if fig_name is None:
        fig_name = os.path.join(exper.output_dir, config.T_dist_fig_name + "_" + data_set + config.dflt_plot_ext)
    if save:
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if show:
        plt.show()
    plt.close()


def plot_qt_probs(exper, data_set="train", fig_name=None, height=16, width=12, save=False, show=False, plot_idx=[]):
    bar_width = 0.5
    if data_set == "train":
        T = len(exper.epoch_stats["qt_hist"])
        opt_step_hist = exper.epoch_stats["opt_step_hist"]
        qt_hist = exper.epoch_stats["qt_hist"]
        plot_title = "Training - q(t|T) distributions for different T (mean={})".format(int(exper.avg_num_opt_steps))
    else:
        T = len(exper.val_stats["qt_hist"])
        opt_step_hist = exper.val_stats["opt_step_hist"]
        qt_hist = exper.val_stats["qt_hist"]
        plot_title = "Validation - q(t|T) distributions "

    res_qts = OrderedDict()
    for i in range(1, T + 1):
        if opt_step_hist[i - 1] != 0:
            res_qts[i] = qt_hist[i] * 1. / opt_step_hist[i - 1]
        else:
            res_qts[i] = []

    if len(plot_idx) == 0:
        plot_idx = [int(exper.avg_num_opt_steps + i) for i in range(-config.qt_mean_range, config.qt_mean_range+1)]
        if data_set == 'val':
            # for now we make only one plot for the validation statistics because we fixed the # of steps to 4
            # determine number of plots we can make
            if np.sum(opt_step_hist > 5) == 1:
                plot_idx = [np.argmax(opt_step_hist) + 1]
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
                if idx not in res_qts:
                    plot_idx.remove(idx)

            num_of_plots = len(plot_idx)
    else:
        num_of_plots = len(plot_idx)

    fig = plt.figure(figsize=(width, height))
    fig.suptitle(plot_title, **config.title_font)
    for i in range(1, num_of_plots + 1):
        index = range(1, plot_idx[i - 1] + 1)
        if i == 1:
            ax1 = plt.subplot(num_of_plots, 2, i)
        else:
            _ = plt.subplot(num_of_plots, 2, i, sharey=ax1)
        plt.bar(index, res_qts[plot_idx[i - 1]], bar_width, color='b', align='center')
        plt.xticks(index)
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