import numpy as np
import torch
import torch.nn as nn
import math
import os
import torch.nn.functional as F
import cv2
import random

from learning_agent.common_utils import identity
from utils.utils import softmax_stable, get_device, to_one_hot, custom_action_encoding
from learning_agent.architectures.cnn import CNNLayer, CNN, Conv2d_MLP_Model, VAE, VQVAE
from learning_agent.architectures.mlp import MLP, Embed, Linear
from lstm.lstm import LSTM
from lstm.network import Network
from scipy.special import softmax
from lstm.attention import SelfAttentionForRL, Attention
from lstm.film import FiLM
#from lstm.ConvLSTM import ConvLSTMNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

experiment_name = 'MiniGrid_rudder'
logger_project_name = 'rudder'

class RRLSTM(nn.Module):
    def __init__(self, config):
        super(RRLSTM, self).__init__()
        self.config = config
        self.cnn_autoencoder = False
        #Pre_LSTM_Linear layer
        #self.pre_linear_layer = Linear(self.config["REWARD_LEARNING"]["input_size"], self.config["REWARD_LEARNING"]["prelinear_embedding"])
        
        #Atenttion model
        if not self.config["REWARD_LEARNING"]["is_lstm"]:
            self.attn_model = SelfAttentionForRL(observation_size = self.config["TRANSFORMER"]["observation_size"],
                                                 action_size = self.config["TRANSFORMER"]["action_size"],
                                                 device = device,
                                                 embedding_size=self.config["TRANSFORMER"]["embedding_size"],
                                                 dim_feedforward=self.config["TRANSFORMER"]["dim_feedforward"],
                                                 pad_val=self.config["TRANSFORMER"]["pad_val"],
                                                 max_len=self.config["TRANSFORMER"]["max_len"],
                                                 verbose=False).to(device)
        
        
        #Attention
        #self.attention = Attention(self.config["REWARD_LEARNING"]["n_units"], self.config["REWARD_LEARNING"]["n_units"], self.config["REWARD_LEARNING"]["n_units"])
        
        #Reduection_layer
        #self.embedding = nn.Linear(6400, 512)
        
        #Constructing the LSTM layer
        in_channels = [self.config["REWARD_LEARNING"]["input_size"]] + [self.config["REWARD_LEARNING"]["n_units"]]*(self.config["REWARD_LEARNING"]["n_layers"]-1)
        self.lstm_layers = nn.Sequential()
        for i, channels in enumerate(in_channels):
            lstm_layer = LSTM(fc_input_size = channels,
                              n_units = self.config["REWARD_LEARNING"]["n_units"])
            self.lstm_layers.add_module("lstm_{}".format(i), lstm_layer)
        #self.lstm_layers = nn.LSTM(self.config["REWARD_LEARNING"]["input_size"], self.config["REWARD_LEARNING"]["n_units"], 1, bidirectional = True)
        #LSTM_Linear layer
        self.linear_layer = Linear(self.config["REWARD_LEARNING"]["n_units"]//2, 1)
        
        #Auxilary task layer
        self.aux_layer = Linear(self.config["REWARD_LEARNING"]["n_units"]//2, 1)
               
        #Constructing attention layer
        #self.attention = nn.Linear(self.config["REWARD_LEARNING"]["n_units"], 1)
        
        #FiLM layer
        self.film = FiLM(self.config["REWARD_LEARNING"]["action_embedding_dim"], self.config["REWARD_LEARNING"]["feature_size"])
        
        #Post LSTM linear layers
        self.post_fc1 = Linear(self.config["REWARD_LEARNING"]["n_units"], self.config["REWARD_LEARNING"]["n_units"]//2)
        self.post_fc2 = Linear(self.config["REWARD_LEARNING"]["n_units"], self.config["REWARD_LEARNING"]["n_units"]//2)
        
        #Sigmoid activation
        #self.sigmoid = nn.Sigmoid()
        
        #Relu activation
        self.relu = nn.LeakyReLU(negative_slope=0.02)
        
        #Tanh activation
        #self.tanh = torch.tanh()
        
        #Batch Normalization
        #self.bn1 = nn.BatchNorm1d(self.config["CNN"]["fc_output_size"])
        #self.bn2 = nn.BatchNorm1d(self.config["REWARD_LEARNING"]["input_size"])
        #self.bn3 = nn.BatchNorm1d(self.config["REWARD_LEARNING"]["n_units"])
        
        #Dropout layer
        #self.dropout = nn.Dropout(p=0.5)
        
        #Linear layer to reduce the dimension of the state
        #self.reduction_layer = nn.Linear(self.config["CNN"]["fc_output_size"], self.config["REWARD_LEARNING"]["action_embedding_dim"])
        
        #Action embedding layer
        #self.action_embedding = nn.Linear(self.config["REWARD_LEARNING"]["n_actions"], self.config["REWARD_LEARNING"]["action_embedding_dim"])
        
        # #Constructing the CNN encoder
        # self.cnn_encoder = CNNEncode(self.config["CNN"]["input_channels"], self.config["CNN"]["kernel_sizes"], self.config["CNN"]["strides"])
        
        # #Construting the hidder linear layer
        # self.hidden = HiddenLayer(self.config["CNN"]["fc_hidden_sizes"])
        
        # #Constructing the cnn decoder
        # self.cnn_encoder = CNNDecode(self.config["CNN"]["input_channels"], self.config["CNN"]["kernel_sizes"], self.config["CNN"]["strides"])
        
    
        #Constructing the action embedding layer
        #self.embedding = nn.Embedding(self.config["REWARD_LEARNING"]["n_actions"], self.config["REWARD_LEARNING"]["action_embedding_dim"])
        
        #Conv layer
        # self.cnn_layer = CNNLayer(input_channels = 10,
        #                           output_channels = 10,
        #                           kernel_size = 3,
        #                           stride=1,
        #                           padding=0,
        #                           pre_activation_fn=identity,
        #                           activation_fn=nn.LeakyReLU(),
        #                           post_activation_fn=identity,
        #                           gain = math.sqrt(2))
        
        # self.cnn = CNN([self.cnn_layer], None)
        
        #Aux Linear layer
        #self.aux_linear = Linear(self.config["REWARD_LEARNING"]["n_units"], 1)
        
        #Constructing MLP models
        # self.mlp = MLP(input_size = 170,
        #                output_size = 1,
        #                hidden_sizes = [32],
        #                output_activation = torch.sigmoid,
        #                use_output_layer = True)
        
        #Constructing the network
        #self.network = Network(self.linear_layer, self.lstm_layer, self.attention, )#, self.conv_mlp)# self.embedding)
        
    # def set_dqn_model(self, dqn_model):
    #     self.dqn_model = dqn_model
        
    # def freeze_weight(self, model):
    #     #Freeze the CNN model from updating
    #     for param in model.parameters():
    #         param.requires_grad = False
        
    # def set_convolution_feature_model(self, is_dqn_feature_extractor, dqn_model, is_dqn_weight_clone):
    #     if is_dqn_feature_extractor:
    #         self.set_dqn_model(dqn_model)
    #         self.freeze_weight(self.dqn_model)
    #         self.cnn = self.dqn_model
    #     else:
    #         if is_dqn_weight_clone:
    #             self.set_dqn_model(dqn_model)
    #             self.cnn = self.dqn_model
    #         else:
    #             #Constructing CNN conv model
    #             self.cnn_autoencoder = True
    #             self.cnn = VAE(input_dim = self.config["VAE"]["input_channels"], 
    #                            latent_dim = self.config["VAE"]["latent_dim"],
    #                            encoder_channels = self.config["VAE"]["encoder_channels"],
    #                            encoder_kernel_sizes = self.config["VAE"]["encoder_kernel_sizes"],
    #                            encoder_strides = self.config["VAE"]["encoder_strides"],
    #                            encoder_paddings = self.config["VAE"]["encoder_paddings"],
    #                            encoder_img_dim = self.config["VAE"]["encode_img_dim"],
    #                            decoder_channels = self.config["VAE"]["decoder_channels"],
    #                            decoder_kernel_sizes = self.config["VAE"]["decoder_kernel_sizes"],
    #                            decoder_strides = self.config["VAE"]["decoder_strides"],
    #                            decoder_paddings = self.config["VAE"]["decoder_paddings"]).to(device)
                
            
    def forward(self, states, action, train_len):
        #TODO Add attention layer
        # b,t,c,h,w = states.unsqueeze(2).size()
        # states = states.view(b*t,c,h,w)
        # if self.cnn_autoencoder:
        #     x_hat, states_features, mean, std = self.cnn(states)
        # else:
        #     _, states_features = self.cnn(states)
        # b,t,f = states.size()
        # states = states.view(b*t, -1)
        # if b*t !=1:
        #     states = self.bn1(states)
        # states = torch.relu(states)
        #states_features = states_features.view(b,t,-1)
        #states = self.dropout(states)
        #states_embedded = self.reduction_layer(states)
        #states_embedded = states_embedded.view(b,t,-1)
        action_one_hot = custom_action_encoding(action, self.config["REWARD_LEARNING"]["n_actions"], self.config["REWARD_LEARNING"]["action_embedding_dim"])
        action_one_hot = torch.tensor(action_one_hot).to(device)
        if self.config["REWARD_LEARNING"]["is_FiLM"]:
            lstm_input = self.film(action_one_hot, states)
        else:
            lstm_input = torch.cat((states, action_one_hot), dim=-1)
        #lstm_input = lstm_input.view(b*t, -1)
        #if b*t !=1:
        #    lstm_input = self.bn2(lstm_input)
        #lstm_input = torch.relu(lstm_input)
        #lstm_input = self.dropout(lstm_input)
        #lstm_input = lstm_input.view(b,t,-1)
        if self.config["REWARD_LEARNING"]["is_lstm"]:
            lstm_output = self.lstm_layers(lstm_input)
            attn = None
        else:
            lstm_output, attn = self.attn_model(lstm_input, train_len, True)
        #lstm_output = self.lstm_layer2(lstm_output[0])
        #lstm_output = lstm_output[0].view(b*t, -1)
        #if b*t !=1:
        #    lstm_output = self.bn3(lstm_output)
        #lstm_output = torch.relu(lstm_output)
        #lstm_output = self.dropout(lstm_output)
        #lstm_output = lstm_output.view(b,t,-1)
        #lstm_output, attn = self.attn_model(lstm_input, train_len, True)
        #attention_weights = F.softmax(self.attention(lstm_output), dim=1)
        #attended_out = attention_weights * lstm_output
        #lstm_output = self.relu(lstm_output)
        lstm_output = self.relu(lstm_output)
        linear_output = self.post_fc1(lstm_output)
        linear_output = self.relu(linear_output)
        estimated_linear_output = self.post_fc2(lstm_output)
        estimated_linear_output = self.relu(estimated_linear_output)
        q_estimate = self.aux_layer(estimated_linear_output)
        q_values = self.linear_layer(linear_output)
        return q_values, q_estimate, attn
        

class LessonBuffer:
    def __init__(self, config, max_time, size):
        self.config = config
        self.size = size
        self.feature_size = self.config["feature_size"]
        self.max_time = max_time
        # Initializing the list to store the transitions
        self.reset_list(self.size)
        self.next_spot_to_add = 0
        self.next_ind = 0
        self.buffer_is_full = False
        self.samples_since_last_training = 0
        self.min_trajectory_score = 0
        self.index_min_score = 0
        self.alpha = 1
        
        
    def reset_list(self, size):
        self.states_buffer = np.zeros(shape=(size, self.max_time, self.feature_size), dtype=np.float32)
        self.next_states_buffer = np.zeros(shape=(size, self.max_time, self.feature_size), dtype=np.float32)
        self.actions_buffer = np.zeros(shape=(size, self.max_time), dtype=np.float32)
        self.rewards_buffer = np.zeros(shape=(size, self.max_time), dtype=np.float32)
        self.lens_buffer = np.zeros(shape=(size, 1), dtype=np.int32)
        self.trajectory_score = np.zeros(shape=(size, 1), dtype=np.float32)
        self.lstm_loss = np.zeros(shape=(size, 1), dtype=np.float32)
        self.sample_weight = np.full((size,), 1/self.size, dtype=np.float32)
        self.sample_priorities = np.full((size,), 100.0, dtype=np.float32)
        
    def dump_buffer_data(self, dump_dir, mode):
        if  dump_dir != None:
            # if is_random_policy:
            #     np.save(path + '/random/states_buffer', self.states_buffer, allow_pickle=True)
            #     np.save(path + '/random/actions_buffer', self.actions_buffer, allow_pickle=True)
            #     np.save(path + '/random/lens_buffer', self.lens_buffer, allow_pickle=True)
            #     np.save(path + '/random/rewards_buffer', self.rewards_buffer, allow_pickle=True)
            # else:
            if mode == 'both':
                dump_dir += '/both'
            elif mode == 'avoid':
                dump_dir += '/avoid'
            else:
                dump_dir += '/preference'
            np.save(dump_dir + '/states_buffer', self.states_buffer, allow_pickle=True)
            np.save(dump_dir + '/next_states_buffer', self.next_states_buffer, allow_pickle=True)
            np.save(dump_dir + '/actions_buffer', self.actions_buffer, allow_pickle=True)
            np.save(dump_dir + '/lens_buffer', self.lens_buffer, allow_pickle=True)
            np.save(dump_dir + '/rewards_buffer', self.rewards_buffer, allow_pickle=True)
        print("[INFO] DQN Training data saved to ", dump_dir)
                
    def fill_buffer_from_disk(self, load_dir, mode):
        data = os.listdir(load_dir)
        if mode == 'both':
            load_dir += '/both'
        elif mode == 'avoid':
            load_dir += '/avoid'
        else:
            load_dir += '/preference'
        if len(data)==0:
            raise ValueError("The directory is None")
        else:
            self.states_buffer = np.load(load_dir + '/states_buffer.npy')
            self.actions_buffer = np.load(load_dir + '/actions_buffer.npy')
            self.lens_buffer = np.load(load_dir + '/lens_buffer.npy')
            self.rewards_buffer = np.load(load_dir + '/rewards_buffer.npy')
            # for i in range(self.states_buffer.shape[1]):
            #     cv2.imwrite("frame1.png", np.moveaxis(self.states_buffer[0,i,:,:,:], 0, -1))
            num_datapoints = self.size
            self.states_buffer = self.states_buffer[:num_datapoints]
            self.actions_buffer = self.actions_buffer[:num_datapoints]
            self.lens_buffer = self.lens_buffer[:num_datapoints]
            self.rewards_buffer = self.rewards_buffer[:num_datapoints]
        print("[INFO] Data loaded sucessfully from ", load_dir)
        
    # LSTM training does only make sense, if there are sequences in the buffer which have different returns.
    # LSTM could otherwise learn to ignore the input and just use the bias units.
    def different_returns_encountered(self):
        if self.buffer_is_full:
            return np.unique(self.rewards_buffer[..., -1]).shape[0] > 1
        else:
            return len(np.unique(self.rewards_buffer[:self.next_spot_to_add, -1])) > 1

    # We only train if 32 samples are played by a random policy
    def full_enough(self):
        return self.buffer_is_full or self.next_spot_to_add >= self.self.config["min_episode_rudder"] #Minimum number of episodes to start the Rudder training

    # def reset_dict(self):
    #     self.index_dict = {"Good": [0]*self.size,
    #                        "Intermediate": [],
    #                        "Bad": [0]*self.size
    #                        }
    
    def get_trajectory_score(self, lstm_loss, oracle_feedback):
        array = self.rewards_buffer[0:self.next_spot_to_add, -1]
        if array.size != 0:
            array_mean =  np.mean(self.rewards_buffer[0:self.next_spot_to_add, -1])
        else:
            array_mean = 0
        mean_feedback = abs(oracle_feedback-array_mean)
        return lstm_loss[0].item() + mean_feedback
    
    def get_buffer_is_full(self):
        return self.buffer_is_full
    
    # Add a new episode to the buffer
    def add(self, previous_state, next_state, actions, rewards):
        traj_length = previous_state.shape[0]
        self.next_ind = self.next_spot_to_add
        self.next_spot_to_add = self.next_spot_to_add + 1
        if self.next_spot_to_add >= self.size:
            self.buffer_is_full = True
        #if not self.buffer_is_full:
            #self.next_spot_to_add = self.next_spot_to_add % self.size
        self.states_buffer[self.next_ind, :traj_length] = previous_state
        self.states_buffer[self.next_ind, traj_length:] = 0
        self.next_states_buffer[self.next_ind, :traj_length] = next_state
        self.next_states_buffer[self.next_ind, traj_length:] = 0
        self.actions_buffer[self.next_ind, :traj_length] = actions
        self.actions_buffer[self.next_ind, traj_length:] = 0
        #self.rewards_buffer[self.next_ind, -1] = rewards[-1]
        #Above line was the original line, but the below line were added to train the additional baselines (env reward+LSTM reward).
        self.rewards_buffer[self.next_ind, :traj_length] = rewards
        self.rewards_buffer[self.next_ind, traj_length:] = 0
        self.lens_buffer[self.next_ind] = traj_length
        
    # def calculate_prob_list(self):
    #     max_length = len(self.index_dict["Bad"]) + len(self.index_dict["Good"])
    #     good_prob =len(self.index_dict["Bad"])/max_length
    #     bad_prob = len(self.index_dict["Good"])/max_length
    #     weight_list = [0]*max_length
    #     for indices in self.index_dict["Good"]:
    #        weight_list[indices] = good_prob
    #     for indices in self.index_dict["Bad"]:
    #        weight_list[indices] = bad_prob
    #     return softmax_stable(weight_list)
    # Choose <batch_size> samples uniformly at random and return them.
    # def sample(self, batch_size):
    #     self.samples_since_last_training = 0
    #     if self.buffer_is_full:
    #         indices = np.random.choice(np.array(range(self.size)), batch_size, 
    #                                    p = self.calculate_prob_list())
    #     else:
    #         indices = np.random.choice(np.array(range(self.next_spot_to_add)), batch_size, 
    #                                     p = self.calculate_prob_list())
    #     return (self.states_buffer[indices, :, :], self.actions_buffer[indices, :],
    #             self.rewards_buffer[indices, :], self.lens_buffer[indices, :])
    
    def get_rank(self, array):
        order = array[:,0].argsort()
        return order.argsort()

    def get_probaility(self, array):
        array = np.sum(array, 1)
        unique = np.unique(array, return_index=True, return_inverse=True, return_counts=True)
        if len(unique[0] != 0):
            prob = 1-softmax(unique[3])
        else:
            prob = softmax(unique[3])
        array_prob = np.zeros_like(array)
        array_prob = prob[unique[2]]
        # array_prob_ = softmax(array_prob)
        # unique_ = np.unique(array_prob_, return_index=True, return_inverse=True, return_counts=True)
        return array_prob
    
    def update_priorities(self, index, loss):
        self.sample_priorities[index] = loss
        
    def update_alpha(self, episode):
        self.alpha = (self.config["n_update"] - episode)/self.config["n_update"]
        
    def update_probability(self):
        order_sample_weight = np.argsort(self.sample_priorities)[::-1]
        rank_sample_weight = order_sample_weight.argsort()+1
        def get_prob(x):
            return (1/x)**self.alpha/sum((1/x)**self.alpha)
        probability = get_prob(rank_sample_weight)
        return probability
        
    def do_post_update_works(self, indices, loss, lstm_update):
        self.update_alpha(lstm_update)
        self.update_priorities(indices, loss)
        self.sample_weight = self.update_probability()
        
    def sample(self, batch_size):
        self.samples_since_last_training = 0
        #indices = np.random.choice(range(self.size), size = batch_size, p = self.sample_weight)
        indices = np.random.choice(range(self.size), size = batch_size)
        return (self.states_buffer[indices, :, :], self.actions_buffer[indices, :],
                self.rewards_buffer[indices, :], self.lens_buffer[indices, :], indices)
        
    def __len__(self):
        return self.next_spot_to_add


def nograd(t):
    return t.detach()
