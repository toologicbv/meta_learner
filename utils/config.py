
class MetaConfig(object):

    def __init__(self):

        # used as filename for the logger
        self.logger_filename = "run.log"
        # directory for models
        self.model_path = "models"
        # file extension for model files
        self.save_ext = '.pkl'
        # directory for figures
        self.figure_path = 'figures'
        # directory for data e.g. validation data
        self.data_path = 'data'
        # filename for validation data
        self.validation_funcs = 'validation_functions.dll'
        # directory for logs, actually every run creates a new directory under this dir
        self.log_root_path = 'logs'
        # prefix for directory under logs for each run
        self.exper_prefix = 'run_'
        # colormap for gradient path
        self.color_map = 'summer'
        # filename that stores Experiment object dumped with dill
        self.exp_file_name = "exp_statistics.dll"
        # filename of figure for loss
        self.loss_fig_name = "loss.png"
        # filename of figure for parameter error (MSE)
        self.param_error_fig_name = "param_error.png"
        # filename of figure for ACT loss
        self.act_loss_fig_name = "loss_act.png"
        # bar-plot of the distribution of optimization steps T
        self.T_dist_fig_name = "T_dist.png"
        # bar-plot of q(t|T) distributions
        self.qt_dist_prefix = "qt_T_dist"
        self.qt_mean_range = 3
        # probability of continue computation
        self.continue_prob = 0.8
        # horizon for maximum number of timesteps
        self.T = 30

        # some fonts defaults for headers of plots
        self.title_font = {'fontname': 'Arial', 'size': '14', 'color': 'black', 'weight': 'normal'}


config = MetaConfig()
