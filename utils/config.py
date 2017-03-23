
class MetaConfig(object):

    def __init__(self):

        self.logger_filename = "run.log"
        self.model_path = "models"
        self.save_ext = '.pkl'
        self.figure_path = 'figures'
        self.data_path = 'data'
        self.validation_funcs = 'validation_functions.dll'
        self.log_root_path = 'logs'
        self.exper_prefix = 'run_'
        self.color_map = 'summer'
        self.exp_file_name = "exp_statistics.dll"
        self.loss_fig_name = "loss.png"
        self.param_error_fig_name = "param_error.png"

config = MetaConfig()