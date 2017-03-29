from config import config
import matplotlib.pyplot as plt

import os
import numpy as np


def loss_plot(exper, fig_name=None, loss_type="normal", height=8, width=6, save=False, show=False):
    title_font = {'fontname': 'Arial', 'size': '14', 'color': 'black', 'weight': 'normal'}
    plt.figure(figsize=(height, width))
    plt.xlabel("epochs")

    if loss_type == "normal":
        if exper.args.learner == "act":
            num_opt_steps = exper.avg_num_opt_steps
        else:
            num_opt_steps = exper.args.optimizer_steps

        train_loss = exper.epoch_stats['loss']
        val_loss = exper.val_stats['loss']
        plt.ylabel("loss")
        if fig_name is None:
            fig_name = os.path.join(exper.output_dir, config.loss_fig_name)
    elif loss_type == "act":
        num_opt_steps = exper.avg_num_opt_steps
        train_loss = exper.epoch_stats['act_loss']
        val_loss = exper.val_stats['act_loss']
        plt.ylabel("act-loss")
        if fig_name is None:
            fig_name = os.path.join(exper.output_dir, config.act_loss_fig_name)
    else:
        raise ValueError("loss_type {} not supported".format(loss_type))
    x_vals = range(1, exper.epoch+1, 1)
    plt.plot(x_vals, train_loss, 'r', label="train-loss")

    if not exper.args.learner == "act":
        if len(x_vals) <= 10:
            plt.xticks(x_vals)
        x_vals = [i for i in range(exper.epoch+1) if i % exper.args.eval_freq == 0]
        if exper.args.eval_freq == 1:
            x_vals = x_vals[1:]
        else:
            x_vals[0] = 1
            if x_vals[-1] != exper.epoch - 1:
                x_vals.append(exper.epoch)
            else:
                x_vals[-1] = exper.epoch

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

    title_font = {'fontname': 'Arial', 'size': '14', 'color': 'black', 'weight': 'normal'}
    plt.figure(figsize=(height, width))
    plt.xlabel("epochs")
    plt.ylabel("parameter error")
    x_vals = range(1, exper.epoch+1, 1)
    plt.plot(x_vals, exper.epoch_stats['param_error'], 'r', label="train-loss")

    if not exper.args.learner == "act":
        if len(x_vals) <= 10:
            plt.xticks(x_vals)
        x_vals = [i for i in range(exper.epoch + 1) if i % exper.args.eval_freq == 0]
        if exper.args.eval_freq == 1:
            x_vals = x_vals[1:]
        else:
            x_vals[0] = 1
            if x_vals[-1] != exper.epoch - 1:
                x_vals.append(exper.epoch)
            else:
                x_vals[-1] = exper.epoch
        plt.plot(x_vals, exper.val_stats['param_error'], 'b', label="valid-loss")
    plt.legend(loc="best")
    plt.title("Train/validation param-error {}-epochs/{}-opt-steps/{}-loss-func".format(exper.epoch,
                                                                                 exper.avg_num_opt_steps,
                                                                                 exper.args.loss_type), **title_font)
    if fig_name is None:
        fig_name = os.path.join(exper.output_dir, config.param_error_fig_name)
    if save:
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if show:
        plt.show()
    plt.close()


def plot_histogram(exper, fig_name=None, height=8, width=6, save=False, show=False):
    title_font = {'fontname': 'Arial', 'size': '14', 'color': 'black', 'weight': 'normal'}
    bar_width = 0.5
    # because we shift the distribution by 1 to start with t=1 until config.T we also need to increase the
    # indices here
    index = np.arange(1, len(exper.opt_step_hist) + 1)
    norms = 1. / np.sum(exper.opt_step_hist) * exper.opt_step_hist
    o_mean = int(round(np.sum(index * norms)))
    plt.figure(figsize=(height, width))
    plt.bar(index, norms, bar_width, color='b', label="p(T) distribution (q={:.3f})".format(config.continue_prob))
    # plot mean value again...in red
    plt.bar([o_mean], norms[o_mean + 1], bar_width, color='r')
    plt.xlabel("optimization steps")
    plt.ylabel("probability")
    plt.title("Distribution of optimization steps T (E[T|{}]={})".format(config.T, o_mean), **title_font)
    plt.legend(loc="best")
    if fig_name is None:
        fig_name = os.path.join(exper.output_dir, config.T_dist_fig_name)
    if save:
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if show:
        plt.show()
    plt.close()
