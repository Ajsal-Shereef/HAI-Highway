import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm
from utils.utils import write_log
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from learning_agent.architectures.mlp import MLP
from preference_learning.reward_model import PEBBLE
from preference_learning.preference_feedback import FeedBack

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

log_keys = ["PEBBLE Epoch", "PEBBLE Loss"]

class TrainingHandler():
    def __init__(self, config, lstm_buffer, dump_dir, logger):
        self.buffer = lstm_buffer
        self.config = config
        self.logger = logger
        self.dump_dir = dump_dir
        n_actions = config["REWARD_LEARNING"]["n_actions"]
        action_embedding_dim = config["REWARD_LEARNING"]["action_embedding_dim"]
        self.model = PEBBLE(self.config)
        self._init_network(self.config)
        self.feedback = FeedBack(config["REWARD_LEARNING"]["size"])
        self.loss = nn.BCELoss()
        
    def _init_network(self, config):
        optim_params = self.config["Optim"]
        #Create the optimizer#
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=optim_params["lstm_lr"],
            weight_decay=optim_params["l2_regularization"],
            eps=optim_params["adam_eps"],
        )
        
    def get_model(self):
        return self.model
    
    def check_training_conditions(self, episode, training_delay):
        if (self.lstm_buffer.different_returns_encountered() and self.lstm_buffer.full_enough()):
            # Samples will be drawn from the lessons buffer.
            if episode % training_delay == 0:
                self.n_lstm_update += 1
                return True
        else:
            return False
        
    def loss_function(self, input, target):
        return self.loss(input, target)
    
    def update_model(self, loss, model, optimizer, retain_graph=False):
        optimizer.zero_grad()
        loss.backward(retain_graph = retain_graph)
        clip_grad_norm_(model.parameters(), 0.50)
        optimizer.step()
    
    def get_relative_probability(self, reward_traje_1, reward_traje_2, length):
        """This function calculates the P_psi defined in the paper https://arxiv.org/pdf/2106.05091.pdf"""
        # Create the mask to nullify the reward value of the padded sequence
        self.mask1 = torch.squeeze(torch.zeros_like(reward_traje_1))
        self.mask2 = torch.squeeze(torch.zeros_like(reward_traje_2))
        for l_num, l in enumerate(length):
            self.mask1[l_num, :l[0]] = 1
            self.mask2[l_num, :l[1]] = 1
        #Multiplying with the self.mask to avoid the padded sequence
        reward_traje_1 = torch.squeeze(reward_traje_1) * self.mask1
        reward_traje_2 = torch.squeeze(reward_traje_2) * self.mask2
        #Finding the reward sum of two trajectory sequence
        reward_sum1 = torch.sum(reward_traje_1, 1)
        reward_sum2 = torch.sum(reward_traje_2, 1)
        max = torch.max(torch.max(reward_sum1, reward_sum2))
        prob = torch.exp(reward_sum1-max)/(torch.exp(reward_sum1-max) + torch.exp(reward_sum2-max))
        return prob
    
    def sample(self, trajectory_states, trajectory_actions, feedback_array, length_array, batch_size):
        #indices = np.random.choice(range(self.size), size = batch_size, p = self.sample_weight)
        indices = np.random.choice(range(len(feedback_array)), size = batch_size)
        return trajectory_states[indices, ...], trajectory_actions[indices, ...], feedback_array[indices, ...], length_array[indices, ...], indices
        
    def train(self):
        states, actions, human_feedback, lens, indices = self.buffer.lstm_reply_buffer.sample(self.config["REWARD_LEARNING"]["size"])
        self.trajectory_states, self.trajectory_actions, self.feedback_array, self.length_array = self.feedback.get_preference_feedback(states, actions, human_feedback, lens)
        self.trajectory_states = torch.tensor(self.trajectory_states).to(device)
        self.trajectory_actions = torch.tensor(self.trajectory_actions).to(device)
        self.feedback_array = torch.tensor(self.feedback_array).float().to(device)
        self.length_array = torch.tensor(self.length_array)
        #Create a pytorch dataloader here to make batchwise training
        #train_dataloader = DataLoader((self.trajectory_pair, self.feedback_array, self.length_array), 10, shuffle=True)
        if self.config["REWARD_LEARNING"]['is_load_lstm']:
            self.model.load_state_dict(torch.load(self.config["REWARD_LEARNING"]['model_dir'])['pebble_weight'])
            print("[INFO] PEBBLE model loaded from ", self.config["REWARD_LEARNING"]['model_dir'])
        else:
            update = 0
            n_updates = 2000#self.config["REWARD_LEARNING"]["n_update"]
            pbar_lstm = tqdm(total=n_updates)
            while update < n_updates:
                trajectory_states, trajectory_actions, feedback_array, length_array, indices = self.sample(self.trajectory_states, self.trajectory_actions, self.feedback_array, self.length_array, self.config["REWARD_LEARNING"]["batch_size"])
                reward_traje_1, _, _ = self.model(trajectory_states[:,0,...], trajectory_actions[:,0,...], length_array[0])
                reward_traje_2, _, _ = self.model(trajectory_states[:,1,...], trajectory_actions[:,1,...], length_array[1])
                prob = self.get_relative_probability(reward_traje_1, reward_traje_2, length_array)
                loss = self.loss_function(prob, feedback_array)
                self.update_model(loss, self.model, self.optimizer)
                #Dumping the PEBBLE model
                if update % 1000 == 0:
                    checkpoint = {"pebble_weight" : self.model.state_dict()}
                    path = self.dump_dir + '/pebble_{}.tar'.format(update)
                    torch.save(checkpoint, path)
                log_value = [update, loss.detach().item()]
                write_log(self.logger, log_keys, log_value, None)
                pbar_lstm.set_description("Loss {}".format(loss.detach().item()))
                pbar_lstm.update(1)
                update += 1
            pbar_lstm.close()    
        
    