from config import config
import matplotlib.pyplot as plt
from pylab import MaxNLocator
import cmocean

import os
import numpy as np
from scipy.stats import geom
from collections import OrderedDict
from datetime import datetime
from pytz import timezone
from itertools import cycle
from probs import ConditionalTimeStepDist
from regression import neg_log_likelihood_loss, nll_with_t_dist


def create_exper_label(exper):

    retrain = "_retrain" if exper.args.retrain else ""
    if exper.args.learner[0:6] != "act_sb" and exper.args.learner != "act_graves":
        label1 = exper.args.learner + exper.args.version + "_" + str(exper.args.max_epoch) + "ep_" + \
            str(int(exper.avg_num_opt_steps)) + "ops" + retrain
    elif exper.args.learner == "act_graves":
        label1 = exper.args.learner + exper.args.version + "_" + str(exper.args.max_epoch) + "ep_" + \
                 "tau{:.5}".format(exper.config.tau) + retrain
    else:
        label1 = exper.args.learner + exper.args.version + "_" + str(exper.args.max_epoch) + "ep_" + \
                 "nu{:.3}".format(exper.config.ptT_shape_param) + retrain

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
    return np.array(x_vals)


def get_exper_loss_data(exper, loss_type, fig_name=None, validation=False):
    exper_label = "_" + create_exper_label(exper)
    dt = "_" + datetime.now(timezone('Europe/Berlin')).strftime('%Y-%m-%d %H:%M:%S.%f')[-15:-7]
    if loss_type == "loss":
        if exper.args.learner == "act":
            num_opt_steps = exper.avg_num_opt_steps
        else:
            num_opt_steps = exper.args.optimizer_steps

        train_loss = exper.epoch_stats['loss']
        val_loss = exper.val_stats['loss']
        plt.ylabel("loss (optimizee)")
        loss_fig_name = exper.config.loss_fig_name

    elif loss_type == "opt_loss":
        num_opt_steps = exper.avg_num_opt_steps
        train_loss = exper.epoch_stats['opt_loss']
        val_loss = exper.val_stats['opt_loss']
        plt.ylabel("optimizer-loss")
        loss_fig_name = exper.config.opt_loss_fig_name

    elif loss_type == "param_error":
        num_opt_steps = exper.avg_num_opt_steps
        train_loss = exper.epoch_stats['param_error']
        val_loss = exper.val_stats['param_error']
        plt.ylabel("Parameter error")
        loss_fig_name = exper.config.param_error_fig_name
    else:
        raise ValueError("loss_type {} not supported".format(loss_type))
    if fig_name is None:
        run_type = "_eval" if validation else "_train"
        fig_name = os.path.join(exper.output_dir, loss_fig_name + run_type + exper_label + config.dflt_plot_ext)
    return num_opt_steps, train_loss, val_loss, fig_name


def loss_plot(exper, fig_name=None, loss_type="loss", height=8, width=6, save=False, show=False, validation=True,
              log_scale=False, only_val=False):

    title_font = {'fontname': 'Arial', 'size': '14', 'color': 'black', 'weight': 'normal'}
    ax = plt.figure(figsize=(height, width)).gca()
    num_opt_steps, train_loss, val_loss, fig_name = get_exper_loss_data(exper, loss_type, fig_name=fig_name,
                                                                        validation=validation)
    plt.xlabel("epochs")
    x_vals = np.arange(1, exper.epoch+1, 1)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    if not only_val:
        if log_scale:
            plt.semilogy(x_vals[2:].astype(int), train_loss[2:], 'r', label="train-loss")
        else:
            plt.plot(x_vals[2:].astype(int), train_loss[2:], 'r', label="train-loss")

    if len(x_vals) <= 10:
        plt.xticks(x_vals.astype(int))

    if validation:
        x_vals = create_x_val_array(exper, val_loss)

    if len(x_vals) > 2:
        offset = 1
    else:
        offset = 0

    if validation:
        if log_scale:
            plt.semilogy(x_vals[offset:].astype(int), val_loss[offset:], 'b', label="valid-loss")
        else:
            plt.plot(x_vals[offset:].astype(int), val_loss[offset:], 'b', label="valid-loss")

    plt.legend(loc="best")
    p_title = "Train/validation loss for model {} (epochs {})".format(exper.args.learner+exper.args.version,
                                                                      exper.args.max_epoch)
    if exper.args.learner[0:6] != "act_sb" and exper.args.learner != "act_graves":
        p_title += " ({}-opt-steps)".format(num_opt_steps)
    if (exper.args.learner == "meta" and exper.args.version == "V3.1") or \
            (exper.args.learner == "act" and exper.args.version == "V2") or \
            (exper.args.learner[0:6] == "act_sb"):
        p_title += r' ($\nu = {:.2}$)'.format(exper.config.ptT_shape_param)
    elif exper.args.learner == "act_graves":
        p_title += r' ($\tau = {:.5}$)'.format(exper.config.tau)
    plt.title(p_title, **title_font)

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


def plot_dist_optimization_steps(exper, data_set="train", fig_name=None, height=8, width=10, save=False, show=False,
                                 epoch=None):
    if epoch is None:
        epoch = exper.epoch

    title_font = {'fontname': 'Arial', 'size': '14', 'color': 'black', 'weight': 'normal'}
    bar_width = 0.5

    if data_set == "train":
        epoch_keys = exper.epoch_stats["opt_step_hist"].keys()
        stats_dict = exper.epoch_stats["opt_step_hist"]
        if exper.args.learner[0:6] == "act_sb" or exper.args.learner == "act_graves":
            epoch_keys = exper.epoch_stats["halting_step"].keys()
            opt_step_hist = exper.epoch_stats["halting_step"][epoch]
        T = exper.config.T
    else:
        if exper.args.learner[0:6] == "act_sb" or exper.args.learner == "act_graves":
            epoch_keys = exper.val_stats["halting_step"].keys()
            opt_step_hist = exper.val_stats["halting_step"][epoch]
        else:
            epoch_keys = exper.val_stats["opt_step_hist"].keys()
            stats_dict = exper.val_stats["opt_step_hist"]
        T = exper.config.max_val_opt_steps

    if exper.args.learner[0:6] != "act_sb" and exper.args.learner != "act_graves":
        for e, epoch_key in enumerate(epoch_keys):

            if e == 0:
                opt_step_hist = np.zeros(len(stats_dict[epoch_key]))
                opt_step_hist += stats_dict[epoch_key]
            else:
                opt_step_hist += stats_dict[epoch_key]

    # because we shift the distribution by 1 to start with t=1 until config.T we also need to increase the
    # indices here
    index = range(1, len(opt_step_hist) + 1)
    norms = 1. / np.sum(opt_step_hist) * opt_step_hist
    o_mean = int(round(np.sum(index * norms)))
    model = "Model {} - ".format(exper.args.learner + exper.args.version)
    if exper.args.learner[0:6] != "act_sb" and exper.args.learner != "act_graves":
        p_title = model + " Distribution of number of optimization steps (E[T|{}]={})".format(T, o_mean)
        p_label = "with p(T|nu)={:.3f})".format(config.pT_shape_param)
        y_label = "Proportion"
    else:
        if exper.args.learner[0:6] == "act_sb":
            p_title = model + r" Histogram of halting step (" + data_set + \
                      r" in epoch {}) with prior $p(t|\nu={:.3f}$)".format(epoch, exper.config.ptT_shape_param)
        if exper.args.learner == "act_graves":
            p_title = model + r" Histogram of halting step (" + data_set + \
                      r" in epoch {}) with $\tau={:.5f}$)".format(epoch, exper.config.tau)
        p_label = ""
        y_label = "Frequencies"
        maxT = np.max(opt_step_hist.nonzero())
        index = range(1, maxT + 1)
        norms = opt_step_hist[:maxT]

    plt.figure(figsize=(width, height))
    plt.bar(index, norms, bar_width, color='b', align='center',
            label=p_label)
    # plot mean value again...in red
    if exper.args.learner[0:6] != "act_sb" and exper.args.learner != "act_graves":
        plt.bar([o_mean], norms[o_mean - 1], bar_width, color='r', align="center")
    plt.xlabel("Number of optimization steps")
    plt.ylabel(y_label)
    plt.title(p_title, **title_font)
    if exper.args.learner[0:6] != "act_sb" and exper.args.learner != "act_graves":
        plt.legend(loc="best")
    if fig_name is None:
        fig_name = os.path.join(exper.output_dir, exper.config.T_dist_fig_name + "_" + data_set +
                                exper.config.dflt_plot_ext)
    else:
        fig_name = os.path.join(exper.output_dir, fig_name + exper.config.dflt_plot_ext)

    if save:
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)

    if show:
        plt.show()
    plt.close()


def plot_qt_probs(exper, data_set="train", fig_name=None, height=16, width=12, save=False, show=False, plot_idx=[],
                  plot_prior=False, epoch=None, add_info=False):
    """
    NOTE: this procedure can be used for plotting the qt distribution for the meta and act models but not for the
    ACT_SB model!!!
    :param exper:
    :param data_set:
    :param fig_name:
    :param height:
    :param width:
    :param save:
    :param show:
    :param plot_idx:
    :param plot_prior:
    :param epoch:
    :param add_info:
    :return:
    """
    if epoch is None:
        epoch = exper.epoch

    bar_width = 0.3
    if data_set == "train":
        T = len(exper.epoch_stats["qt_hist"][epoch])
        opt_step_hist = exper.epoch_stats["opt_step_hist"][epoch]
        qt_hist = exper.epoch_stats["qt_hist"][epoch]
        plot_title = "Training " + exper.args.problem + r"- q(t|T) $\nu={:.2f}$ (E[T]={})".format(exper.config.ptT_shape_param,
            int(exper.avg_num_opt_steps))
        if not exper.args.fixed_horizon:
            plot_title += " (stochastic training)"
    else:
        T = len(exper.val_stats["qt_hist"][epoch])
        opt_step_hist = exper.val_stats["opt_step_hist"][epoch]
        qt_hist = exper.val_stats["qt_hist"][epoch]
        plot_title = exper.args.problem + r" - $q(t|{}) \;\; \nu={:.2}$".format(
            config.max_val_opt_steps, exper.config.ptT_shape_param)
        if not exper.args.fixed_horizon:
            plot_title += " (stochastic training E[T]={})".format(int(exper.avg_num_opt_steps))
    res_qts = OrderedDict()
    for i in range(1, T + 1):
        if opt_step_hist[i - 1] != 0:
            res_qts[i] = qt_hist[i] * 1. / opt_step_hist[i - 1]
        else:
            pass

    if len(plot_idx) == 0:
        plot_idx = [int(exper.avg_num_opt_steps + i) for i in range(-config.qt_mean_range, config.qt_mean_range+1)]
        # in case idx=1 in list remove, because probs for 1 step are always 1.
        if 1 in plot_idx:
            plot_idx.remove(1)

        if data_set == 'val':
            num_of_plots = 1
            height = 10
            width = 8
            plot_idx = [len(opt_step_hist)]

        else:
            # training, always enough values at our hand
            check = plot_idx[:]
            for idx in check:
                # if we don't have any results for this number of steps than remove from list
                if idx not in res_qts.keys():
                    plot_idx.remove(idx)

            num_of_plots = len(plot_idx)
    else:
        num_of_plots = len(plot_idx)

    fig = plt.figure(figsize=(height, width))
    fig.suptitle(plot_title, **config.title_font)

    for i in range(1, num_of_plots + 1):
        index = range(1, plot_idx[i - 1] + 1)
        T = len(index)
        if i == 1 and num_of_plots == 1:
            ax1 = plt.subplot(num_of_plots, 1, i)
        elif i == 1:
            ax1 = plt.subplot(num_of_plots, 2, i)
        else:
            _ = plt.subplot(num_of_plots, 2, i, sharey=ax1)

        plt.bar(index, res_qts[plot_idx[i - 1]], bar_width, color='b', align='center',
                label=r"$q(t|{}) \;\; \nu={:.2}$".format(T, exper.config.ptT_shape_param))
        if plot_prior:
            kl_prior_dist = ConditionalTimeStepDist(T=exper.config.max_val_opt_steps,
                                                    q_prob=exper.config.ptT_shape_param)
            # print(exper.config.max_val_opt_steps, exper.config.ptT_shape_param)
            priors = kl_prior_dist.pmfunc(np.arange(1, exper.config.max_val_opt_steps+1))
            # print(priors[0:10])
            plt.bar(np.array(index)+bar_width, priors, bar_width, color='orange', align='center',
                    label=r"prior $p(t|{}) \;\; \nu={:.2}$".format(T, exper.config.ptT_shape_param))
        if data_set == "train":
            if len(index) > 15:
                index = np.arange(1, len(index), 5)
            plt.xticks(index)
        plt.legend(loc="best")
        if i == num_of_plots or i == num_of_plots - 1:
            plt.xlabel("Steps")
        if i % 2 == 0:
            # TODO hide yticks for this subplots, looks better
            pass
        else:
            plt.ylabel("Probs")
        plt.xlim([0, exper.config.max_val_opt_steps])

    if add_info:
        # determine max mode step. Add "1" because the argmax determines index and not step
        idx_mode_probs = np.argmax(exper.val_stats["qt_funcs"][plot_idx[-1]], 1) + 1
        print("Key ", plot_idx[-1])
        N = exper.val_stats["qt_funcs"][plot_idx[-1]].shape[0]
        mean_mode = np.mean(idx_mode_probs)
        median = np.median(idx_mode_probs)
        stddev_mode = np.std(idx_mode_probs)
        min_mode = np.min(idx_mode_probs)
        max_mode = np.max(idx_mode_probs)
        stats = r"Max mode stats(N={}): Range({}, {}) / mean={:.1f} / median={} / stddev={:.1f}".format(N, min_mode,
                                                                                                        max_mode,
                                                                                                        mean_mode,
                                                                                                        median,
                                                                                                        stddev_mode)
        plt.annotate(stats,
                     xy=(0.5, 0), xytext=(0, 0),
                     xycoords=('axes fraction', 'figure fraction'),
                     textcoords='offset points',
                     size=12, ha='center', va='bottom')

    if fig_name is None:
        fig_name = "_" + data_set + "_" + create_exper_label(exper)
        fig_name = os.path.join(exper.output_dir, config.qt_dist_prefix + fig_name + ".png")
    if save:
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if show:
        plt.show()
    plt.close()


def plot_loss_over_tsteps(expers, height=8, width=12, do_show=True, do_save=False, fig_name=None, plot_best=False,
                         loss_type='param_error', min_step=None, max_step=None, sort_exper=None, log_scale=True,
                         with_stddev=True, runID=None, y_lim=[13, 60]):
    # extra_labels = ["", ""]
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

    if min_step is None:
        min_step = 0
    if max_step is not None:
        max_step += 1

    for e in np.arange(num_of_expers):
        if loss_type == "param_error":
            res_dict = expers[e].val_stats["step_param_losses"]
            y_label = "avg final param error"
            plot_title = sort_exper + " (avg parameter error per step)"
        else:
            res_dict = expers[e].val_stats["step_losses"]
            y_label = "avg final loss"
            plot_title = sort_exper + " (avg loss per step)"
        if plot_best:
            plot_title = "Best run selected --- " + plot_title

        keys = res_dict.keys()
        # res = res_dict[keys[len(keys) - i]]
        model = expers[e].args.model
        if "act" in model:
            if expers[e].args.fixed_horizon:
                model += "(fixed-H)"
        if expers[e].args.learner == "act" or (expers[e].args.learner == "meta"
                                               and expers[e].args.version == "V3.1"):
            model += r"($\nu={:.2f}$)".format(expers[e].config.ptT_shape_param)
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
            if runID is None:
                val_run = keys[-1]
            else:
                val_run = runID

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
        if max_step is None:
            max_step = len(res_dict[best_val_runs[e]])

        index = np.arange(min_step, max_step)
        icolor = iter_colors.next()
        if with_stddev:
            losses = expers[e].val_stats["loss_funcs"][:, min_step:max_step]
            mean_losses = np.mean(losses, 0)
            std_losses = np.std(losses, 0)
            mean_plus_std = mean_losses + std_losses
            mean_min_std = mean_losses - std_losses
            y_min_value = np.min(mean_min_std)
            y_max_value = np.max(mean_plus_std)

        else:
            mean_losses = expers[e].val_stats["step_losses"][val_run][min_step:max_step]
            y_min_value = np.min(mean_losses)
            y_max_value = np.max(mean_losses)
        y_min_value -= y_min_value * 0.1
        y_max_value += y_max_value * 0.1
        if plot_best:
            l_label = "{}({})(stop={})".format(model, best_val_runs[e], stop_step)
        else:
            if expers[e].args.learner == "act":
                l_label = "{}(stop={})".format(model, stop_step)
            else:
                l_label = "{}".format(model) # + extra_labels[e]

        if log_scale:
            # res_dict[best_val_runs[e]][min_step:max_step]
            # print(index.shape, mean_losses.shape)
            plt.semilogy(index, mean_losses, color=icolor, dashes=iter_styles.next(),
                         linewidth=2., label=l_label)
            if with_stddev:
                plt.fill_between(index, mean_plus_std, mean_min_std, color=icolor, alpha='0.1')
                plt.yscale("log")
        else:
            plt.plot(index, mean_losses, color=icolor,
                         dashes=iter_styles.next(),
                         linewidth=2., label=l_label)
            if with_stddev:
                plt.fill_between(index, mean_plus_std, mean_min_std, color=icolor, alpha='0.2')

        # if min_step == 0:
        #     start = 1
        # else:
        #    start = min_step
        if len(index) > 150:
            index = np.arange(min_step, max_step + 1, 50)
        elif len(index) > 41:
            index = np.arange(min_step, max_step +1, 10)

        plt.xlim([min_step, max_step+1])
        plt.ylim([y_lim[0], y_lim[1]])
        plt.xticks(index)
        plt.xlabel("Number of optimization steps")
        plt.ylabel(y_label)
        plt.legend(loc="best")
        plt.title(plot_title, **title_font)

    if do_save:
        # dt = "_" + datetime.now(timezone('Europe/Berlin')).strftime('%Y-%m-%d %H:%M:%S.%f')[-15:-7]
        # dt = str.replace(str.replace(dt, ':', '_'), '-', '')
        dt = ""
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
    plot_title += " loss per epoch"
    plt.figure(figsize=(width, height))
    plt.title(plot_title, **title_font)
    style = [[8, 4, 2, 4, 2, 4], [4, 2, 2, 4, 2, 4], [2, 2, 2, 4, 2, 4], [4, 8, 2, 4, 2, 4], [8, 8, 8, 4, 2, 4],
             [4, 4, 2, 8, 2, 2]]
    iter_colors = cycle(p_colors)

    for e in np.arange(num_of_expers):

        if loss_type == "loss":
            loss = "loss"
        elif loss_type == 'opt_loss':
            loss = 'opt_loss'
        elif loss_type == 'param_error':
            loss = "param_error"
        else:
            raise ValueError("{} loss type is not supported".format(loss_type))

        num_opt_steps, train_loss, val_loss, _ = get_exper_loss_data(expers[e], loss, validation=validation)

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


def plot_parm_loss_steps(expers, height=25, width=15, num_of_plots=0, do_show=False, fig_name=None, do_save=False,
                         log_scale=True, loss_type="param_error", max_step=None):
    num_of_expers = len(expers)
    num_of_val_runs = len(expers[0].val_stats["step_param_losses"])

    if num_of_plots == 0:
        num_of_plots = num_of_val_runs

    max_steps = len(expers[0].val_stats["step_param_losses"][expers[0].val_stats["step_param_losses"].keys()[0]])
    index = np.arange(1, max_steps + 1)

    plot_title = "Final average "
    fig = plt.figure(figsize=(width, height))
    if loss_type == "param_error":
        plot_title += "parameter error ({})".format("regression")
        y_label = "Final error"
    else:
        plot_title += "loss ({})".format("regression")
        y_label = "Final loss"
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
            plt.ylabel(y_label)
        if num_of_plots > 2:
            fig.set_figheight(25)
        iter_colors = cycle(p_colors)
        iter_styles = cycle(style)
        for e in np.arange(num_of_expers):
            if loss_type == "param_error":
                res_dict = expers[e].val_stats["step_param_losses"]
            else:
                res_dict = expers[e].val_stats["step_losses"]

            keys = res_dict.keys()
            # could be that not all experiments have the same number of val runs
            if i <= len(keys):
                res = res_dict[keys[len(keys) - i]]
                model = expers[e].args.model
                if max_step is None:
                    max_step = len(res)
                # plt.bar(index + (e*bar_width), res,  bar_width,  color=bar_colors[e], log=True,
                #        align="center", label="{}({})".format(model, keys[len(keys)-i]))
                # plt.xticks(index + bar_width / 2, index-1)
                if log_scale:
                    plt.semilogy(index[0:max_step], res[0:max_step], color=iter_colors.next(),
                                 dashes=iter_styles.next(), linewidth=2.,
                                label="{}({})".format(model, keys[len(keys) - i]))
                else:
                    plt.plot(index[0:max_step], res[0:max_step], color=iter_colors.next(),
                             dashes=iter_styles.next(), linewidth=2.,
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


def plot_kl_div_parts(exper, data_set="val", fig_name=None, height=16, width=12, save=False, show=False,
                      final_terms=False, log_qt=False, plot_prior=False):

    assert data_set == "val", "Dataset {} not yet supported".format(data_set)
    bar_width = 0.3
    plot_title = "Separate KL-divergence components per optimization step"
    fig = plt.figure(figsize=(width, height))
    fig.suptitle(plot_title, **config.title_font)
    index = np.arange(1, exper.config.max_val_opt_steps + 1)
    last_run = sorted(exper.val_stats["ll_loss"].keys())[-1]
    ll_loss = exper.val_stats["ll_loss"][last_run]
    kl_div = exper.val_stats["kl_div"][last_run]
    kl_entropy = exper.val_stats["kl_entropy"][last_run]

    if final_terms:
        qt_probs = np.exp(exper.val_stats["kl_entropy"][last_run])
        ll_loss *= qt_probs
        kl_div *= qt_probs
        kl_entropy *= qt_probs
    ax1 = plt.subplot(2, 1, 1)
    ax1.bar(index, ll_loss, bar_width, color='b', align='center', label="neg-ll")
    ax1.bar(np.array(index) + bar_width, kl_div, bar_width, color='r',
            align='center', label="log p(t|{})".format(exper.config.max_val_opt_steps))
    ax1.set_xlabel("Number of optimization steps")
    ax1.set_ylabel("log probability")
    if log_qt:
        if not final_terms:
            qt_probs = np.exp(exper.val_stats["kl_entropy"][last_run])
        ax1.bar(np.array(index) + 2 * bar_width, kl_entropy, bar_width, color='g',
                align='center', label="log q(t|{})".format(exper.config.max_val_opt_steps))

    ax1.legend(loc="best")
    if log_qt:
        ax2 = plt.subplot(2, 1, 2)
        ax2.set_title("Approximated q(t) distribution")
        ax2.bar(index, qt_probs, bar_width, color='b', align='center', label="q(t|{})".format(exper.config.max_val_opt_steps))
        if plot_prior:
            kl_prior_dist = ConditionalTimeStepDist(T=exper.config.max_val_opt_steps, q_prob=exper.config.ptT_shape_param)
            priors = kl_prior_dist.pmfunc(np.arange(1, exper.config.max_val_opt_steps+1))
            ax2.bar(np.array(index) + bar_width, priors, bar_width, color='orange', align='center',
                    label="prior p(t|{})".format(exper.config.max_val_opt_steps))

        ax2.legend(loc="best")
        ax2.set_xlabel("Number of optimization steps")
        ax2.set_ylabel("probability")

    if fig_name is None:
        fig_name = "_" + data_set + "_" + create_exper_label(exper)
        fig_name = os.path.join(exper.output_dir, "kl_parts" + fig_name + ".png")
    else:
        fig_name = os.path.join(exper.output_dir, fig_name + ".png")
    if save:
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if show:
        plt.show()
    plt.close()


def plot_qt_mode_hist(exper, do_save=False, do_show=False, width=10, height=7, fig_name=None, add_info=True):

    key = exper.val_stats["qt_funcs"].keys()[-1]
    print("key ", key)
    # need to add "1" to all indices which in fact reflect the optimization step when q-prob has highest value (mode)
    max_idx_probs = np.argmax(exper.val_stats["qt_funcs"][key], 1) + 1
    N = exper.val_stats["qt_funcs"][key].shape[0]
    stats = r"Max mode stats(N={}): Range({}, {}) / mean={:.1f} / median={:.1f} / " \
            r"stddev={:.1f}".format(N, np.min(max_idx_probs), np.max(max_idx_probs), np.mean(max_idx_probs),
                                    np.median(max_idx_probs), np.std(max_idx_probs))

    plt.figure(figsize=(width, height))
    plt.title(r"Distribution of mode-step of $q(t|{})$ with $\nu={:.2f}$".format(exper.config.max_val_opt_steps,
                                                                          exper.config.ptT_shape_param))
    _ = plt.hist(max_idx_probs, bins=key+1, normed=True)
    plt.xlim([0, key+1])

    if add_info:
        plt.annotate(stats,
                     xy=(0.5, 0), xytext=(0, 0),
                     xycoords=('axes fraction', 'figure fraction'),
                     textcoords='offset points',
                     size=12, ha='center', va='bottom')

    if fig_name is None and do_save:
        fig_name = "_" + create_exper_label(exper)
        fig_name = os.path.join(exper.output_dir, "qt_mode_stats" + fig_name + ".png")

    if do_save:
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if do_show:
        plt.show()
    plt.close()


def plot_qt_detailed_stats(exper, funcs, do_save=False, do_show=False, width=18, height=15, threshold=0.85,
                           fig_name=None):

    keys = exper.val_stats["qt_funcs"].keys()
    print("Run with key {}".format(keys[-1]))
    qt_probs = exper.val_stats["qt_funcs"][keys[-1]]

    if exper.args.problem == "regression":
        nll_init = neg_log_likelihood_loss(funcs.y, funcs.y_t(funcs.initial_params),
                                           funcs.stddev, N=funcs.n_samples,
                                           sum_batch=False,
                                           size_average=False)
        nll_min = neg_log_likelihood_loss(funcs.y, funcs.y_t(funcs.true_params),
                                          funcs.stddev, N=funcs.n_samples,
                                          sum_batch=False,
                                          size_average=False)
    else:
        nll_init = nll_with_t_dist(funcs.y, funcs.y_t(funcs.initial_params), N=funcs.n_samples, shape_p=1, scale_p=1.,
                                   avg_batch=False)
        nll_min = nll_with_t_dist(funcs.y, funcs.y_t(funcs.true_params), N=funcs.n_samples, shape_p=1, scale_p=1.,
                                  avg_batch=False)

    nll_distance = (nll_init - nll_min).data.cpu().squeeze().numpy()
    max_idx_probs = np.argmax(qt_probs, 1) + 1
    qt_cdf = np.cumsum(qt_probs, 1)
    N = nll_distance.shape[0]
    # stops = np.searchsorted(qt_cdf[2, :], threshold)
    stops_steps = np.apply_along_axis(lambda a: a.searchsorted(threshold) + 1, axis=1, arr=qt_cdf)

    min_x = np.min(max_idx_probs)
    max_x = np.max(max_idx_probs)
    max_y = np.max(nll_distance)
    max_y += 0.1 * max_y

    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(hspace=.5)
    p_title = exper.args.problem + r' ($\nu={:.2}$) - ${}$ functions'.format(exper.config.ptT_shape_param, N)
    if not exper.args.fixed_horizon:
        p_title += " (stochastic training E[T]={})".format(int(exper.avg_num_opt_steps))
    fig.suptitle(p_title , **config.title_font)
    bins = np.unique(max_idx_probs)
    # print(bins)
    ax1 = plt.subplot(4, 2, 1)
    ax1.set_title("Max mode-step versus NLL distance at step 0 (N={})".format(nll_distance.shape[0]),
                  **config.title_font)
    _ = ax1.scatter(max_idx_probs, nll_distance, s=5, color='g', alpha=0.2)
    ax1.set_xlim([min_x - 1, max_x + 1])
    if len(bins) > 10:
        ax1.set_xticks(np.arange(min_x - 1, max_x + 2), 5)
    else:
        ax1.set_xticks(bins)

    ax1.set_xlabel("Maximum mode-step")
    ax1.set_ylim([0, max_y])
    ax1.set_ylabel("Distance NLL(start)-NLL(min)")

    ax2 = plt.subplot(4, 2, 2)
    mean_mode = np.mean(max_idx_probs)
    ax2.set_title("Distribution max-mode steps (N={}) mean={:.1f}".format(nll_distance.shape[0], mean_mode),
                  **config.title_font)

    if len(bins) == 1:
        bins = np.array([bins[0]-1, bins[0], bins[0]+1])
    _ = ax2.hist(max_idx_probs, bins=bins, align='left', normed=True)
    ax2.set_xlim([min_x - 1, max_x + 1])
    if len(bins) > 10:
        ax2.set_xticks(np.arange(min_x - 1, max_x + 2), 5)
    else:
        ax2.set_xticks(bins)

    ax2.set_xlabel("Maximum mode-step")

    bins = np.unique(stops_steps)
    # print(bins)
    min_x = np.min(stops_steps)
    max_x = np.max(stops_steps)
    ax3 = plt.subplot(4, 2, 3)
    ax3.set_title("Stopping step versus NLL distance at step 0 (N={})".format(nll_distance.shape[0]),
                  **config.title_font)
    _ = ax3.scatter(stops_steps, nll_distance, s=5, alpha=0.2, color="r",
                    label=r'threshold ${:.2f}$'.format(threshold))
    ax3.set_xlim([min_x - 1, max_x + 1])
    if len(bins) > 10:
        ax3.set_xticks(np.arange(min_x - 1, max_x + 2), 5)
    else:
        ax3.set_xticks(bins)
    ax3.set_xlabel("Stopping step")
    ax3.set_ylim([0, max_y])
    ax3.set_ylabel("Distance NLL(start)-NLL(min)")
    ax3.legend(loc="best")

    ax4 = plt.subplot(4, 2, 4)
    mean_step = np.mean(stops_steps)
    ax4.set_title("Distribution stopping steps (N={}) mean={:.1f}".format(nll_distance.shape[0], mean_step),
                  **config.title_font)

    _ = ax4.hist(stops_steps, bins=bins, normed=True, align='left', label=r'threshold ${:.2f}$'.format(threshold))
    ax4.set_xlim([min_x - 1, max_x + 1])
    if len(bins) > 10:
        ax4.set_xticks(np.arange(min_x - 1, max_x + 2), 5)
    else:
        ax4.set_xticks(bins)
    ax4.set_xlabel("Stopping step")
    ax4.legend(loc="best")

    if fig_name is None and do_save:
        fig_name = "_" + create_exper_label(exper) + "_td{:.2f}".format(threshold)
        fig_name = os.path.join(exper.output_dir, "qt_detailed_stats" + fig_name + ".png")

    if do_save:
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)

    if do_show:
        plt.show()

    plt.close()


def plot_image_map_losses(exper, data_set="train", fig_name=None, width=18, height=15, do_save=False, do_show=False,
                          cmap=cmocean.cm.haline, scale=[11, 70]):

    if data_set not in ["train", "eval"]:
        raise ValueError("For parameter -data_set- you can only choose 1)train or 2)eval")

    if exper.args.problem == "regression":
        scale = [9., 70]
    elif exper.args.problem == "mlp":
        scale = [1.5, 2.35]

    if data_set == "train":
        run_type = "training"
        y_label = "Training epoch"
        X = np.vstack(exper.epoch_stats["step_losses"].values())
    else:
        y_label = "Evaluation run"
        run_type = "evaluation (each {} epoch)".format(int(exper.args.eval_freq))
        X = np.vstack(exper.val_stats["step_losses"].values())
    # we can be (nearly) sure to never reach zero loss values (exactly) and because those value disturbe the colormap
    # we set them to a "bad" value in order to color them specifically. This is necessary for the ACT model when
    # trained with a stochastic horizon
    min_value = np.min(X[X > 0])
    X[X == 0] = -1
    plt.figure(figsize=(width, height))
    if exper.args.learner == "act" or (exper.args.learner == "meta" and exper.args.version == "V2"):
        stochastic = r" (stochastic horizon E[T]={} $\nu={:.3f}$)".format(int(exper.avg_num_opt_steps),
                                                                          exper.config.ptT_shape_param)
    elif exper.args.learner[0:6] == "act_sb":
        stochastic = r" ($\nu={:.3f}$)".format(exper.config.ptT_shape_param)
    elif exper.args.learner == "act_graves":
        stochastic = r" ($\tau={:.5f}$)".format(exper.config.tau)
    else:
        stochastic = ""

    ptitle = "Model {} - ".format(exper.args.learner+exper.args.version)
    plt.title(ptitle + "loss per time step during " + run_type + stochastic, **config.title_font)
    # use the combination of "vmin" in imshow and "cmap.set_under()" to label the "bad" (zero) values with a specific
    # color
    cmap.set_under(color='darkgray')
    im = plt.imshow(X, cmap=cmap, interpolation='none', aspect='auto', vmin=scale[0], vmax=scale[1])
    plt.xlabel("Number of optimization steps")
    plt.ylabel(y_label)
    if X.shape[0] > 25:
        y_step_size = X.shape[0] // 10
    else:
        y_step_size = 1
    plt.yticks(np.arange(0, X.shape[0], y_step_size), np.arange(1, X.shape[0]+1, y_step_size))
    plt.colorbar(im)

    if fig_name is None and do_save:
        fig_name = "_" + create_exper_label(exper)
        fig_name = os.path.join(exper.output_dir, data_set + "_step_loss_map" + fig_name + ".png")
    if do_save:
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if do_show:
        plt.show()

    plt.close()


def plot_actsb_qts(exper, data_set="train", fig_name=None, height=16, width=12, save=False, show=False,
                   plot_prior=False, epoch=None, add_info=False):
    if data_set not in ["train", "eval"]:
        raise ValueError("For parameter -data_set- you can only choose 1)train or 2)eval")

    if epoch is None:
        # just the last epoch of the experiment
        epoch = exper.epoch

    bar_width = 0.3
    model_info = "Model {} - ".format(exper.args.learner + exper.args.version)
    if data_set == "train":
        T = np.max(exper.epoch_stats["qt_hist"][epoch].nonzero()) + 1
        qt_hist = exper.epoch_stats["qt_hist"][epoch][:T]
        # print(np.array_str(exper.epoch_stats["qt_hist"][epoch][:T+2], precision=4))
        plot_title = model_info + "q(t|x) during TRAINING for epoch {} (".format(epoch) + exper.args.problem
        if exper.args.learner[0:6] == "act_sb":
            plot_title += r") - with prior(t|$\nu={:.2f})$".format(exper.config.ptT_shape_param)
        elif exper.args.learner == "act_graves":
            plot_title += r") - $\tau={:.5f}$".format(exper.config.tau)
    else:
        T = np.max(exper.val_stats["qt_hist"][epoch].nonzero()) + 1
        qt_hist = exper.val_stats["qt_hist"][epoch][:T]
        plot_title = model_info + "q(t|x) during EVALUATION for epoch {} (".format(epoch) + exper.args.problem
        if exper.args.learner[0:6] == "act_sb":
            plot_title += r") - with prior(t|$\nu={:.2f})$".format(exper.config.ptT_shape_param)
        elif exper.args.learner == "act_graves":
            plot_title += r") - $\tau={:.5f}$".format(exper.config.tau)

    ax = plt.figure(figsize=(height, width)).gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.title(plot_title, **config.title_font)
    index = np.arange(1, qt_hist.shape[0] + 1).astype(int)

    plt.bar(index, qt_hist, bar_width, color='b', align='center',
            label=r"$q(t)$")
    if plot_prior:
        priors = geom.pmf(index, exper.config.ptT_shape_param)
        plt.bar(index+bar_width, priors, bar_width, color='orange', align='center',
                label=r"$p(t|\nu={:.2})$".format(exper.config.ptT_shape_param))
    if data_set == "train":
        if len(index) > 15:
            index = np.arange(1, len(index), 5)
        plt.xticks(index)
    plt.legend(loc="best")
    plt.xlabel("time step")
    plt.ylabel("qt probabilities")

    if add_info:
        # determine max mode step. Add "1" because the argmax determines index and not step
        idx_mode_probs = np.argmax(qt_hist, 1) + 1
        N = qt_hist.shape[0]
        mean_mode = np.mean(idx_mode_probs)
        median = np.median(idx_mode_probs)
        stddev_mode = np.std(idx_mode_probs)
        min_mode = np.min(idx_mode_probs)
        max_mode = np.max(idx_mode_probs)
        stats = r"Max mode stats(N={}): Range({}, {}) / mean={:.1f} / median={} / stddev={:.1f}".format(N, min_mode,
                                                                                                        max_mode,
                                                                                                        mean_mode,
                                                                                                        median,
                                                                                                        stddev_mode)
        plt.annotate(stats,
                     xy=(0.5, 0), xytext=(0, 0),
                     xycoords=('axes fraction', 'figure fraction'),
                     textcoords='offset points',
                     size=12, ha='center', va='bottom')

    if fig_name is None:
        fig_name = "_" + data_set + "_" + create_exper_label(exper)
        fig_name = os.path.join(exper.output_dir, "qt_values" + fig_name + ".png")
    else:
        fig_name = os.path.join(exper.output_dir, fig_name + ".png")
    if save:
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if show:
        plt.show()
    plt.close()


def plot_image_map_data(exper, data_set="train", fig_name=None, width=18, height=15, do_save=False, do_show=False,
                        data="qt_value", cmap=cmocean.cm.haline, scale=[0., 1.]):

    if data_set not in ["train", "eval"]:
        raise ValueError("For parameter -data_set- you can only choose 1)train or 2)eval")
    if data not in ["qt_value", "halting_step"]:
        raise ValueError("Only support parameter values for -data- are 1) qt_value 2) halting_step")

    if data == "qt_value":
        if data_set == "train":
            run_type = "training"
            y_label = "Training epoch"
            X = np.vstack(exper.epoch_stats["qt_hist"].values())
        else:
            y_label = "Evaluation run"
            run_type = "evaluation (each {} epoch)".format(int(exper.args.eval_freq))
            X = np.vstack(exper.val_stats["qt_hist"].values())

        title_prefix = "Model {} - qt probabilities during ".format(exper.args.learner+exper.args.version) + run_type
        save_suffix = "qts"
        scale[0] = 0.
        scale[1] = 1.

    elif data == "halting_step":
        if data_set == "train":
            run_type = "training"
            y_label = "Training epoch"
            X = np.vstack(exper.epoch_stats["halting_step"].values())
        else:
            y_label = "Evaluation run"
            run_type = "evaluation (each {} epoch)".format(int(exper.args.eval_freq))
            X = np.vstack(exper.val_stats["halting_step"].values())

        title_prefix = "Model {} - halting step during ".format(exper.args.learner+exper.args.version) + run_type
        save_suffix = "halting"
        scale[0] = np.min(X[X>0])
        # print("min {}".format(scale[0]))
        scale[1] = np.max(X)

    # we can be (nearly) sure to never reach zero loss values (exactly) and because those value disturbe the colormap
    # we set them to a "bad" value in order to color them specifically. This is necessary for the ACT model when
    # trained with a stochastic horizon
    min_value = np.min(X[X > 0])
    X[X == 0] = -1
    plt.figure(figsize=(width, height))
    if exper.args.learner[0:3] == "act":
        p_title = title_prefix + " per time step" + \
                     r" ($\nu={:.3f}$)".format(exper.config.ptT_shape_param)
    else:
        p_title = ""
    if exper.args.learner == "act_graves":
        p_title = title_prefix + " per time step" + \
                  r" ($\tau={:.5f}$)".format(exper.config.tau)

    plt.title(p_title, **config.title_font)
    # use the combination of "vmin" in imshow and "cmap.set_under()" to label the "bad" (zero) values with a specific
    # color
    cmap.set_under(color='darkgray')
    im = plt.imshow(X, cmap=cmap, interpolation='none', aspect='auto', vmin=scale[0], vmax=scale[1])
    plt.xlabel("Number of optimization steps")
    plt.ylabel(y_label)
    if X.shape[0] > 25:
        y_step_size = X.shape[0] // 10
    else:
        y_step_size = 1
    plt.yticks(np.arange(0, X.shape[0], y_step_size), np.arange(1, X.shape[0]+1, y_step_size))
    plt.colorbar(im)

    if fig_name is None and do_save:
        fig_name = "_" + create_exper_label(exper)
        fig_name = os.path.join(exper.output_dir, data_set + "_step_map_" + save_suffix + fig_name + ".png")
    if do_save:
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if do_show:
        plt.show()

    plt.close()


def plot_halting_step_stats_with_loss(exper, height=8, width=12, do_show=False, do_save=False,
                                      fig_name=None, add_info=True):

    if exper.args.problem == "regression_T":
        opt_loss_lim = [20., 60.]
        kl_lim = None
        avg_step_lim = [0., 65.]
    else:
        opt_loss_lim = None
        kl_lim = None
        avg_step_lim = None

    fig, ax = plt.subplots()
    fig.set_figheight(height)
    fig.set_figwidth(width)
    suffix = " (KL cost annealing)" if (exper.args.learner[0:6] == "act_sb" and exper.args.version == "V2") else ""
    ptitle = "Model {} - ".format(exper.args.learner+exper.args.version) + \
             "halting step statistics versus loss components during training "
    if exper.args.learner == "act_graves":
        ptitle += r" ($\tau={:.5f}$)".format(exper.config.tau)
    else:
        ptitle += r" ($\nu={:.3f}$)".format(exper.config.ptT_shape_param)

    plt.title(ptitle + suffix, **config.title_font)
    opt_hist = exper.epoch_stats["opt_step_hist"]
    epochs = len(opt_hist)

    halting_stats = np.vstack(exper.epoch_stats["halting_stats"].values())
    halt_min = halting_stats[:, 0]
    halt_max = halting_stats[:, 1]
    halt_avg = halting_stats[:, 2]
    halt_median = halting_stats[:, 4]
    halt_stddev = halting_stats[:, 3]
    halt_avg_plus_stddev = halt_avg + halt_stddev
    halt_avg_min_stddev = halt_avg - halt_stddev
    kl_terms = exper.epoch_stats["kl_term"]
    # duration = exper.epoch_stats["duration"]
    opt_loss = exper.epoch_stats["opt_loss"]

    x = np.arange(1, epochs + 1)
    l1, = ax.plot(x, opt_loss, marker="D", label="optimizer loss", c='r')
    ax.set_ylabel("optimizer loss")
    if opt_loss_lim is not None:
        ax.set_ylim(opt_loss_lim)
    # second y-axis
    ax2 = ax.twinx()
    if exper.args.learner == "act_graves":
        graph_label = "penalty-term"
    else:
        graph_label = "kl-divergence"
    l2, = ax2.plot(x, kl_terms, marker='o', label=graph_label, c='g')
    if exper.args.learner == "act_graves":
        ax2.set_ylabel("penalty-term")
    else:
        ax2.set_ylabel("kl-divergence")
    if kl_lim is not None:
        ax2.set_ylim(kl_lim)

    ax3 = ax.twinx()
    l3, = ax3.plot(x, halt_avg, marker='*', label="average halting step", c='silver')
    ax3.fill_between(x, halt_avg_min_stddev, halt_avg_plus_stddev, color='silver', alpha='0.2')
    ax3.spines["right"].set_position(("axes", 1.2))
    ax3.set_ylabel("average halting step")
    lines = [l1, l2, l3]
    ax3.legend(lines, [l.get_label() for l in lines], loc="best")
    if avg_step_lim is not None:
        ax3.set_ylim(avg_step_lim)

    ax.set_xlabel("Epochs")

    if add_info:
        idx = exper.args.max_epoch - 1
        stats = r"Halting step statistics final epoch: Range({}, {}) / mean={:.1f} / stddev={:.1f} / " \
                r"median={:.1f}".format(int(halt_min[idx]), int(halt_max[idx]), halt_avg[idx], halt_stddev[idx],
                                        halt_median[idx])
        plt.annotate(stats,
                     xy=(0.5, 0), xytext=(0, 0),
                     xycoords=('axes fraction', 'figure fraction'),
                     textcoords='offset points',
                     size=12, ha='center', va='bottom')

    if do_save:
        if fig_name is None:
            fig_name = "halting_step_stats"
        fig_name = os.path.join(exper.output_dir, fig_name + config.dflt_plot_ext)

        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)

    if do_show:
        plt.show()

    plt.close()


def plot_loss_versus_halting_step(exper, height=8, width=12, do_show=False, do_save=False,
                                  fig_name=None, epoch=None, x_max=None):
    if epoch is None:
        epoch = exper.args.max_epoch

    halting_steps = exper.val_stats["halt_step_funcs"][epoch]
    nll_distance = exper.val_stats["loss_funcs"][epoch]
    min_x = np.min(halting_steps)
    max_x = np.max(halting_steps)
    # min_y = np.min(nll_distance)
    max_y = np.max(nll_distance)
    fig, ax = plt.subplots()
    fig.set_figheight(height)
    fig.set_figwidth(width)
    p_title = "Model {} - halting step versus NLL distance at step 0 (N={}) during evaluation ".format(
        exper.args.learner + exper.args.version, nll_distance.shape[0]
    )
    ax.set_title(p_title + "(epoch={})".format(epoch), **config.title_font)
    _ = ax.scatter(halting_steps, nll_distance, s=5, alpha=0.2, color="r",
                   label=r" ($\nu={:.3f}$)".format(exper.config.ptT_shape_param))
    if x_max is None:
        ax.set_xlim([min_x - 1, max_x + 1])
    else:
        ax.set_xlim([min_x - 1, x_max])
    ax.set_xlabel("Halting step")
    ax.set_ylim([0, max_y])
    ax.set_ylabel("Distance NLL(start)-NLL(min)")
    ax.legend(loc="best")

    if do_save:
        if fig_name is None:
            fig_name = "halting_step_versus_nll_distance"
        fig_name = os.path.join(exper.output_dir, fig_name + config.dflt_plot_ext)

        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)

    if do_show:
        plt.show()

    plt.close()


def plot_gradient_stats(exper, height=8, width=12, do_show=False, do_save=False, fig_name=None, offset=1):

    fig, ax = plt.subplots()
    fig.set_figheight(height)
    fig.set_figwidth(width)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    model_name = exper.args.learner + exper.args.version
    grad_means = exper.epoch_stats["grad_stats"][:, 0]
    grad_std = exper.epoch_stats["grad_stats"][:, 1]
    grad_max = grad_means + grad_std
    grad_min = grad_means - grad_std
    x = np.arange(1, grad_means.shape[0] + 1)
    # ax.errorbar(x, grad_means, yerr=grad_std, fmt='-o')
    ax.fill_between(x[offset:], grad_min[offset:], grad_max[offset:], color='silver', alpha='0.2')
    ax.plot(x[offset:], grad_means[offset:], 'o-', color="black")
    ax.set_title("Model {} - gradient statistics during training (batch-size={})".format(model_name, exper.args.batch_size),
                 **config.title_font)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("mean value gradients")
    if do_save:
        if fig_name is None:
            fig_name = "gradient_statistics"
        fig_name = os.path.join(exper.output_dir, fig_name + config.dflt_plot_ext)

        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)

    if do_show:
        plt.show()

    plt.close()