
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
        self.val_file_name_suffix = 'val_funcs_'
        # directory for logs, actually every run creates a new directory under this dir
        self.log_root_path = 'logs'
        # prefix for directory under logs for each run
        self.exper_prefix = 'run_'
        # colormap for gradient path
        self.color_map = 'summer'
        # filename that stores Experiment object dumped with dill
        self.exp_file_name = "exp_statistics.dll"
        """
            Plotting defaults
        """
        # number of functions to plot during validation
        self.num_val_plots = 4
        # default plot file extension
        self.dflt_plot_ext = ".png"
        # filename of figure for loss
        self.loss_fig_name = "loss"
        # filename of figure for parameter error (MSE)
        self.param_error_fig_name = "param_error"
        # filename of figure for ACT loss
        self.act_loss_fig_name = "loss_act"
        # bar-plot of the distribution of optimization steps T
        self.T_dist_fig_name = "T_dist"
        # bar-plot of q(t|T) distributions
        self.qt_dist_prefix = "qt_T_dist"
        self.qt_mean_range = 3
        """
            Important parameter for prior distribution P(T)
        """
        # probability of continue computation
        # self.continue_prob = 0.8  # T=10 => 5ops
        # self.continue_prob = 0.915
        # self.continue_prob = 0.53  # T=10 => 2ops
        # self.continue_prob = 0.66
        # self.continue_prob = 0.73
        # self.continue_prob = 0.86
        # self.continue_prob = 0.88  # T=25 => 10ops
        # self.continue_prob = 0.93
        # self.continue_prob = 0.94
        # self.continue_prob = 0.955
        self.continue_prob = 0.965
        # horizon for maximum number of timesteps
        # self.T = 10 # avg = 5 with 0.82
        # self.T = 20 # avg = 7 with 0.86 or 5ops with 0.8
        # self.T = 20 # avg = 9 with 0.9
        # self.T = 25 # avg = 10 with 0.88
        # self.T = 30 # avg = 15 with 0.94
        # self.T = 40 # avg = 20 with 0.96 or 15 with 0.925
        # self.T = 45 # avg = 20 with 0.95
        # self.T = 60 # avg = 27 with 0.96
        # self.T = 75  # avg = 27 with 0.98 or avg = 10 with 0.86
        self.T = 1000 # avg = 80 with 0.98
        self.max_val_opt_steps = 100

        # some fonts defaults for headers of plots
        self.title_font = {'fontname': 'Arial', 'size': '14', 'color': 'black', 'weight': 'normal'}

        """
            Stop threshold
        """
        self.qt_threshold = 0.95

config = MetaConfig()
