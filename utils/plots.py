from config import config
import matplotlib.pyplot as plt

import os


def loss_plot(exper, fig_name=None, height=8, width=6, save=False, show=False):

    plt.figure(figsize=(height, width))
    plt.xlabel("epochs")
    plt.ylabel("loss")
    x_vals = range(1, exper.epoch+1, 1)
    plt.plot(x_vals, exper.epoch_stats['loss'], 'r', label="train-loss")
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

    plt.plot(x_vals, exper.val_stats['loss'], 'b', label="valid-loss")
    plt.legend(loc="best")
    plt.title("Train/validation loss {}-epochs/{}-opt-steps/{}-loss-func".format(exper.epoch,
                                                                                 exper.args.optimizer_steps,
                                                                                 exper.args.loss_type))
    if fig_name is None:
        fig_name = os.path.join(exper.output_dir, config.loss_fig_name)
    if save:
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if show:
        plt.show()
    plt.close()


def param_error_plot(exper, fig_name=None, height=8, width=6, save=False, show=False):

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
    else:
        x_vals[0] = 1
        if x_vals[-1] != exper.epoch - 1:
            x_vals.append(exper.epoch)
        else:
            x_vals[-1] = exper.epoch
    plt.plot(x_vals, exper.val_stats['param_error'], 'b', label="valid-loss")
    plt.legend(loc="best")
    plt.title("Train/validation param-error {}-epochs/{}-opt-steps/{}-loss-func".format(exper.epoch,
                                                                                 exper.args.optimizer_steps,
                                                                                 exper.args.loss_type))
    if fig_name is None:
        fig_name = os.path.join(exper.output_dir, config.param_error_fig_name)
    if save:
        plt.savefig(fig_name, bbox_inches='tight')
        print("INFO - Successfully saved fig %s" % fig_name)
    if show:
        plt.show()
    plt.close()
