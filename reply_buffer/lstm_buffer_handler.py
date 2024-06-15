import numpy as np
from copy import deepcopy
from lstm.convo_lstm_model import LessonBuffer
from utils.utils import to_one_hot, flatten_state, dump_list, standardize


class LSTMBufferHandler():
    """This class handles the LSTM reply buffer"""
    def __init__(self, config, max_time, size):                            
        self.reset_arrays()
        self.max_time = max_time
        self.config = config["REWARD_LEARNING"]
        self.lstm_reply_buffer = LessonBuffer(self.config, self.max_time, size)
        self.mean_feedback = 0
        if self.config["save_trajectory"]:
            self.good_traj = []
            self.bad_traj = []
        
    def add_transitions(self, previous_state, 
                        action, reward, next_state, 
                        key_state):
        #previous_state = flatten_state(previous_state_encoding)
        #next_state = flatten_state(next_state_encoding)
        self.previous_states.append(previous_state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.key_state.append(key_state)
        
    def get_trajectory(self):        
        return self.previous_states, self.actions, self.rewards, self.next_states
    
    def get_state_encoding(self):
        return self.states, self.actions
    
    def set_dump_load_dir(self, dump_dir, load_dir):
        self.dump_dir = dump_dir
        self.load_dir = load_dir
    
    def reset_arrays(self):
        self.previous_states = []
        self.actions = []
        self.next_states = []
        self.rewards = []
        self.key_state = []
        
    def reset_list(self, size):
        self.lstm_reply_buffer.reset_list(size)
        
    def dump_buffer_data(self, dump_dir, mode):
        self.lstm_reply_buffer.dump_buffer_data(dump_dir, mode)
        
    def fill_buffer_from_disk(self, dump_dir, mode):
        self.lstm_reply_buffer.fill_buffer_from_disk(dump_dir, mode)
        
    def get_trajectory_score(self, lstm_loss, oracle_feedback):
        return self.lstm_reply_buffer.get_trajectory_score(lstm_loss, oracle_feedback)
    
    def get_is_buffer_is_full(self):
        self.is_reply_buffer_is_full = self.lstm_reply_buffer.get_buffer_is_full()
    
    def add_to_buffer(self, feedback):
        previous_state = np.stack(self.previous_states)
        next_state = np.stack(self.next_states)
        actions = np.array(self.actions)
        rewards = np.array(self.rewards)
        #This below line 68 has commented for easing the training of additional baselines. This will dump the env reward trajectories.
        rewards[-1] = feedback
        self.lstm_reply_buffer.add(previous_state=previous_state, next_state = next_state, actions=actions, rewards=rewards)
                 
    def add_to_buffer_from_list(self, list, feedback):
        list[2][-1] = feedback
        self.lstm_reply_buffer.add(np.array(list[0]), np.array(list[1]), np.array(list[2]))
        
    def dump_trajectory(self, feedback):
        if feedback == 1 and len(self.good_traj) < 100:
           self.good_traj.append([self.previous_states, self.actions,
                                  self.rewards, self.next_states, 
                                  self.dones, self.info])
        elif feedback == -1 and len(self.bad_traj) < 100:
           self.bad_traj.append([self.previous_states, self.actions,
                                  self.rewards, self.next_states, 
                                  self.dones, self.info])
        dump_list(self.good_traj, self.config["path_to_dump_traj"] + "/good.pkl")
        dump_list(self.bad_traj, self.config["path_to_dump_traj"] + "/bad.pkl")