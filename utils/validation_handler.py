import os
import time
import numpy as np

import torch
from torch.autograd import Variable

from utils.common import generate_fixed_weights, get_func_loss
from utils.helper import preprocess_gradients


class ValidateMLPOnMetaLearner(object):

    def __init__(self, exper, save_model=True):
        self.max_eval_steps = exper.config.max_val_opt_steps
        self.total_opt_loss = 0.
        self.total_loss = 0.
        self.save_model = save_model
        self.avg_final_step_loss = 0.

    def __call__(self, exper, meta_learner, optimizees, with_step_acc=False, verbose=False):
        start_validate = time.time()
        fixed_weights = generate_fixed_weights(exper, steps=exper.config.max_val_opt_steps)
        avg_accuracies = []
        num_of_mlps = len(optimizees)
        for i, mlp in enumerate(optimizees):
            if exper.args.cuda:
                mlp = mlp.cuda()
            meta_learner.reset_lstm(keep_states=False)
            meta_learner.reset_losses()
            mlp.reset()
            col_losses = []
            for step in range(self.max_eval_steps):
                loss = get_func_loss(exper, mlp, average=False)
                loss.backward()
                avg_loss = loss.data.cpu().squeeze().numpy()[0].astype(float)
                col_losses.append(avg_loss)
                input = Variable(torch.cat((preprocess_gradients(mlp.get_flat_grads().data), mlp.get_flat_params().data),
                                           1), volatile=True)
                delta_params = meta_learner.forward(input)
                par_new = mlp.get_flat_params() + delta_params.unsqueeze(1)
                mlp.set_eval_obj_parameters(par_new)
                image, y_true = exper.dta_set.next_batch(is_train=True)
                loss_step = mlp.evaluate(image, use_copy_obj=True, compute_loss=True, y_true=y_true)
                if with_step_acc:
                    accuracy = mlp.test_model(exper.dta_set, exper.args.cuda, quick_test=True)
                    exper.val_stats["step_acc"][exper.epoch][step] += accuracy
                meta_learner.losses.append(Variable(loss_step.data.unsqueeze(1)))
                mlp.set_parameters(par_new)
                mlp.zero_grad()
                del delta_params
                del par_new

            # make another step to register final loss
            loss = get_func_loss(exper, mlp, average=False)
            if with_step_acc:
                accuracy = mlp.test_model(exper.dta_set, exper.args.cuda, quick_test=True)
                exper.val_stats["step_acc"][exper.epoch][step] += accuracy
            # collect results
            col_losses.append(loss.data.cpu().squeeze().numpy()[0].astype(float))
            self.avg_final_step_loss += loss.data.cpu().squeeze().numpy()[0].astype(float)
            np_losses = np.array(col_losses)
            self.total_loss += np.sum(np_losses)
            # concatenate losses into a string for plotting the gradient path
            exper.val_stats["step_losses"][exper.epoch] += np_losses

            self.total_opt_loss += meta_learner.final_loss(loss_weights=fixed_weights).data.squeeze()[0]

            accuracy = mlp.test_model(exper.dta_set, exper.args.cuda, quick_test=True)
            avg_accuracies.append(accuracy)
            if verbose:
                exper.meta_logger.info("INFO - Epoch {}: "
                                       "Evaluation - accuracies of last MLP: {:.4f}".format(exper.epoch, accuracy))

        exper.val_stats["step_losses"][exper.epoch] *= 1./float(num_of_mlps)
        exper.val_stats["step_acc"][exper.epoch] *= 1. / float(num_of_mlps)
        self.total_loss *= 1./float(num_of_mlps)
        self.total_opt_loss *= 1. / float(num_of_mlps)
        self.avg_final_step_loss *= 1. / float(num_of_mlps)
        exper.val_stats["loss"].append(self.total_loss)
        exper.val_stats["opt_loss"].append(self.total_opt_loss)

        if exper.val_stats["step_losses"][exper.epoch].shape[0] > 210:
            step_results = exper.val_stats["step_losses"][exper.epoch][-100:]
            exper.meta_logger.info(">>> NOTE: only showing last 100 steps <<<")
        else:
            step_results = exper.val_stats["step_losses"][exper.epoch]
        exper.meta_logger.info("INFO - Epoch {}: "
                               "Evaluation - Final step losses: {}".format(exper.epoch,

                                                                np.array_str(step_results, precision=4)))
        avg_accuracies = np.mean(np.array(avg_accuracies))
        exper.meta_logger.info("INFO - Epoch {}: - Evaluation - average accuracy {:.3f}".format(exper.epoch,
                                                                                                avg_accuracies))

        if with_step_acc:
            exper.meta_logger.info("INFO - Epoch {}: "
                                   "Evaluation - Final step accuracies: {}".format(exper.epoch,
                                                                      np.array_str(exper.val_stats["step_acc"][exper.epoch],
                                                                                   precision=4)))

        duration = time.time() - start_validate
        exper.meta_logger.info("INFO - Epoch {}: Evaluation - elapsed time {:.2f} seconds: ".format(exper.epoch,
                                                                                                    duration))
        exper.meta_logger.info("INFO - Epoch {}: Evaluation - "
                               "Final validation stats: total-step-losses / final-step loss "
                               ": {:.4}/{:.4}".format(exper.epoch, self.total_loss, self.avg_final_step_loss))
        exper.add_duration(duration, is_train=False)

        meta_learner.reset_losses()
        if self.save_model:
            model_path = os.path.join(exper.output_dir, meta_learner.name + "_vrun" + str(exper.epoch) +
                                      exper.config.save_ext)
            torch.save(meta_learner.state_dict(), model_path)
            exper.meta_logger.info("INFO - Successfully saved model parameters to {}".format(model_path))
