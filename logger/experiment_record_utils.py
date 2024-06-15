# experiment_record_utils.py
# -------------------------
# helper functions to record experiment data

from datetime import datetime
import os
import sys
import shutil
import pickle
import numpy as np
import torch
import wandb
import json

DEFAULT_PROJECT_NAME = 'default project'
DEFAULT_IS_USE_WANDB = True


class ExperimentLogger:
    def __init__(self, saved_dir, expr_name, is_add_time_to_name=True, save_trajectories=False):
        if is_add_time_to_name:
            now = datetime.now()
            d = now.strftime('%Y%m%d_%H%M%S')
            expr_name = expr_name + '_' + d
        self.expr_name = expr_name
        self.project_name = DEFAULT_PROJECT_NAME
        self.is_use_wandb = DEFAULT_IS_USE_WANDB
        self.is_save_trajs = save_trajectories

        # create expr dir
        if not os.path.exists(saved_dir):
            os.mkdir(saved_dir)
        self.saved_dir = os.path.join(saved_dir, expr_name)
        os.mkdir(self.saved_dir)

        # create image saved dir
        self.img_dir = os.path.join(self.saved_dir, 'images')
        os.mkdir(self.img_dir)

        # create snapshot saved dir
        self.snapshot_dir = os.path.join(self.saved_dir, 'snapshots')
        os.mkdir(self.snapshot_dir)

        # trajectories holder
        self.basic_trajectories = list()    # traj with basic information
        self.current_basic_trajectory = list()

        # create log file
        self.logfile = open(os.path.join(self.saved_dir, 'log.txt'), 'a')

    def add_trajectories(self, trajs):
        """ Save the basic trajectories """
        for traj in trajs:
            for t in range(len(traj)):
                curr_state = traj[t][0]
                action = traj[t][1]
                reward = traj[t][2]
                next_state = traj[t][3]
                done = traj[t][4]
                info = traj[t][5]
                self.add_transition(curr_state, action, reward, next_state, done, info)

    def add_transition(self, curr_state, action, reward, next_state, done, info):
        """
        Save the transition
        """
        if not self.is_save_trajs:
            return
        self.current_basic_trajectory.append((curr_state, action, reward, next_state, done, info))
        if done:
            self.basic_trajectories.append(self.current_basic_trajectory)
            self.current_basic_trajectory = list()

    def save_models(self, checkpoint, postfix=None, is_snapshot=True, prefix='model'):
        """
        Save current model
        :param checkpoint: the parameters of the models, see example in pytorch's documentation: https://pytorch.org/tutorials/beginner/saving_loading_models.html
        :param is_snapshot: whether saved in the snapshot directory
        :param prefix: the prefix of the file name
        :param postfix: the postfix of the file name (can be episode number, frame number and so on)
        """
        saved_dir = self.snapshot_dir if is_snapshot else self.saved_dir

        if postfix is not None:
            fname = get_unique_fname(saved_dir, prefix + '_' + postfix + '.tar')
        else:
            fname = get_unique_fname(saved_dir, prefix + '_' + self.expr_name + '.tar')
        torch.save(checkpoint, os.path.join(saved_dir, fname))
        
    def load_params(self, path, device, dqn, dqn_target, dqn_optim):
        """Load model and optimizer parameters."""

        params = torch.load(path, map_location=device)
        dqn.load_state_dict(params["dqn_state_dict"])
        dqn_target.load_state_dict(params["dqn_target_state_dict"])
        dqn_optim.load_state_dict(params["dqn_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def save_trajectories_snapshot(self):
        """
        save trajectories to file
        is_separated_file: whether to save each trajectory to its own file
        """
        fname = 'traj_' + self.expr_name + '.pickle'
        fname = get_unique_fname(self.snapshot_dir, fname)
        fname = os.path.join(self.snapshot_dir, fname)
        with open(fname, 'wb') as f_traj:
            pickle.dump(self.basic_trajectories, f_traj, protocol=pickle.HIGHEST_PROTOCOL)

    def save_trajectories(self, is_separated_file=False):
        """
        save trajectories to file
        is_separated_file: whether to save each trajectory to its own file
        """
        if is_separated_file:
            for i in range(len(self.basic_trajectories)):
                fname = 'traj_' + str(i) + '_' + self.expr_name + '.pickle'
                fname = os.path.join(self.saved_dir, fname)
                with open(fname, 'wb') as f_traj:
                    pickle.dump([self.basic_trajectories[i]], f_traj, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            fname = 'traj_' + self.expr_name + '.pickle'
            fname = os.path.join(self.saved_dir, fname)
            with open(fname, 'wb') as f_traj:
                pickle.dump(self.basic_trajectories, f_traj, protocol=pickle.HIGHEST_PROTOCOL)

    def save_human_feedback(self, feedbacks, file_name):
        fname = os.path.join(self.saved_dir, file_name)
        with open(fname, 'wb') as file:
            pickle.dump(feedbacks, file, protocol=pickle.HIGHEST_PROTOCOL)

    def save_human_study_server_data(self, study_data, file_path):
        dir_name = os.path.dirname(file_path)
        # if the directory exists, empty it first
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            os.mkdir(dir_name)
        else:
            os.mkdir(dir_name)
        with open(file_path, 'wb') as file:
            pickle.dump(study_data, file, protocol=pickle.HIGHEST_PROTOCOL)

    def read_human_study_server_data(self, file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        return data

    def save_rgb_image(self, fname, img, img_format='png', step=None, episode=None):
        if step is None:
            ts = str(datetime.now().timestamp()).replace('.', '_')
            step = ts
            episode = 'nan'
        fname = fname + '_' + str(episode) + '_' + str(step) + '.' + img_format
        fname = os.path.join(self.img_dir, fname)
        img.save(fname)

    def copy_file(self, source):
        config_fname = os.path.basename(source)
        saved_fname = os.path.join(self.saved_dir, config_fname)
        shutil.copyfile(source, saved_fname)

    def redirect_output_to_logfile_as_well(self):
        class Logger(object):
            def __init__(self, logfile):
                self.stdout = sys.stdout
                self.logfile = logfile

            def write(self, message):
                self.stdout.write(message)
                self.logfile.write(message)

            def flush(self):
                # this flush method is needed for python 3 compatibility.
                # this handles the flush command by doing nothing.
                # you might want to specify some extra behavior here.
                pass
        sys.stdout = Logger(self.logfile)
        sys.stderr = sys.stdout

    def set_is_use_wandb(self, is_use):
        self.is_use_wandb = is_use

    def set_wandb_project_name(self, name):
        self.project_name = name

    def initialise_wandb(self, tags = None):
        if self.is_use_wandb:
            self.run = wandb.init(project=self.project_name, name=self.expr_name, 
                                  settings=wandb.Settings(start_method="fork"), reinit=True, tags=tags)
            
    def finish_run(self):
        self.run.finish()
        
    def save_config_wandb(self, config):
        """
        The config is assumed to be a nested dict, e.g. {"hyper param": {"learning rate": 0.1}}
        """
        wandb_config = {}
        for key0 in config:
            for key1 in config[key0]:
                param_name = str(key0) + '_' + str(key1)
                param_value = config[key0][key1]
                wandb_config[param_name] = str(param_value)
        if self.is_use_wandb:
            wandb.config.update(wandb_config)

    def log_wandb(self, log_dict, step=None):
        if self.is_use_wandb:
            if step is not None:
                wandb.log(log_dict, step=step)
            else:
                wandb.log(log_dict)

    def watch_wandb(self, model_list):
        if self.is_use_wandb:
            wandb.watch(model_list, log='all', log_freq=1, log_graph=True)

    def open_virtual_display(self):
        from pyvirtualdisplay import Display
        display = Display(visible=0, size=(1080, 608))
        display.start()


def get_unique_fname(file_dir, fname_base):
    name, extension = os.path.splitext(fname_base)
    post_fix = 0
    while True:
        fname = name + '_' + str(post_fix) + extension
        if not os.path.exists(os.path.join(file_dir, fname)):
            return fname
        post_fix += 1






