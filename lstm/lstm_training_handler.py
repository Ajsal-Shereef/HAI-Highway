import torch
import numpy as np
import torchinfo

from lstm.rudder import LSTM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LSTMTrainingHandler():
    def __init__(self, config_file, lstm_buffer, dump_dir, logger):
        self.config = config_file["REWARD_LEARNING"]
        self.lstm_buffer = lstm_buffer.lstm_reply_buffer
        self.is_lstm_hiccup_completed = False
        self.n_lstm_update = 0
        self.model = LSTM(config_file, self.lstm_buffer, dump_dir, logger)
        
    def check_training_conditions(self, episode, training_delay):
        if (self.lstm_buffer.different_returns_encountered() and self.lstm_buffer.full_enough()):
            # Samples will be drawn from the lessons buffer.
            if episode % training_delay == 0:
                self.n_lstm_update += 1
                return True
        else:
            return False
    
    def get_lstm_current_loss(self):
        return self.model.get_current_loss()
    
    def watch_model(self):
        self.model.watch_model()

    def get_model(self):
        return self.model.get_model()

    def is_lstm_training_started(self):
        return self.model.is_lstm_training_started()
        
    def train(self):
        self.model.train()
        
    def get_onedwalk_prediction(self, states, action, rewards):
        return self.model.redistribute_reward(states, action, torch.tensor(rewards).to(device))
        
    def get_rudder_prediction(self, states):
        redistributed_reward = self.model.redistribute_reward(states)
        redistributed_reward = redistributed_reward.tolist()[0]
        return redistributed_reward
    def get_is_lstm_min_hiccup_completed(self):
        return self.is_lstm_hiccup_completed  
