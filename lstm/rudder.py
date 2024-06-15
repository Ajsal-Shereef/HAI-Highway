import gc
import math
import yaml
import torch
import torchinfo
import numpy as np
import torch.optim as optim
import torch.nn.functional as F

from tqdm import tqdm

from torch.nn import MSELoss as MSELoss
from torch.nn.utils import clip_grad_norm_
from logger.neptunelogger import Logger
from utils.utils import *
from scipy.special import softmax
from torch.autograd import Variable
from lstm.convo_lstm_model import RRLSTM
from utils.plot_network_internals import PlotInternal
from logger.experiment_record_utils import ExperimentLogger
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn import metrics
from torch.distributions.normal import Normal
from torch.distributions import kl_divergence
import torchvision.transforms as transforms
from torchvision.ops import sigmoid_focal_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_gpus = 2
log_keys = ["Epoch", "LSTM loss"]
vae_log_keys = ["VAE Epoch", "VAE loss", "VAE Validation loss"]

class LSTM():
    def __init__(self, lstm_config, buffer, dump_dir, logger):
        self.buffer = buffer
        self.config = lstm_config
        self.logger = logger
        self._init_network(lstm_config)
        self.return_scaling = self.config["REWARD_LEARNING"]["return_scaling"]
        self.continuous_pred_factor = self.config["REWARD_LEARNING"]["continuous_pred_factor"]
        self.dump_dir = dump_dir
        self.loss = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.total_loss = 0
        self.plot_internals = PlotInternal()
        #self.batch_norm = torch.nn.BatchNorm1d(1).to(device)
        self.lstm_training_started = False
        # if logger is not None:
        #     logger.watch_wandb(self.model)
        
        
    def _init_network(self, config):
        #Loading the network
        self.model = RRLSTM(config).to(device)
        # if torch.cuda.device_count() > 1:
        #     self.model = torch.nn.DataParallel(self.model)
        self.logger.watch_wandb(self.model)
        #self.model.to(f'cuda:{self.model.device_ids[0]}')
        # torchinfo.summary(self.model, input_size = [(1, 85, 25)],
        #                   device = get_device())
        optim_params = self.config["Optim"]
        #Create the optimizer#
        self.lstm_optimizer = optim.Adam(
            self.model.parameters(),
            lr=optim_params["lstm_lr"],
            weight_decay=optim_params["l2_regularization"],
            eps=optim_params["adam_eps"],
        )
        # self.optimizer = optim.SGD(
        #     self.model.parameters(),
        #     lr=optim_params["lstm_lr"],
        #     weight_decay=optim_params["l2_regularization"],
        #     momentum=0.9,
        # )
        
    def load_cnn_model(self, saved_dir):
        params = torch.load(saved_dir, map_location=device)
        self.cnn.load_state_dict(params["cnn_model_weight"])
        print("[INFO] loaded the CNN model from", saved_dir)
        
    def calculate_reward_loss(self, lstm_out, feedback):
        pred_g0 = torch.cat([torch.zeros_like(lstm_out[:, 0:1, :]), lstm_out], dim=1)[:, :-1, :]
        redistributed_reward = pred_g0[:, 1:, 0] - pred_g0[:, :-1, 0]
        redistributed_reward = redistributed_reward*self.mask[:,1:]
        redistributed_reward = torch.sum(redistributed_reward, dim=1)
        feedback = feedback.squeeze(1)
        reward_loss = F.mse_loss(redistributed_reward, feedback)
        return reward_loss
    
    def calculate_quality(self, lstm_out, oracle_feedback):
        quality_list = []
        pred_g0 = torch.cat([torch.zeros_like(lstm_out[:, 0:1]), lstm_out], dim=1)[:, :-1]
        for batch in range(pred_g0.size(0)):
            last_time_step_difference = abs(pred_g0[batch,-1] - oracle_feedback[batch]).item()
            quality = 1 - (last_time_step_difference/self.config["REWARD_LEARNING"]["mu"])*(1/(1-self.config["REWARD_LEARNING"]["epsilon"]))
            quality_list.append(quality)
        return quality_list
    
    def get_current_loss(self):
        return log_keys, self.log_values
    
    def get_model(self):
        return self.model
    
    def is_lstm_training_started(self):
        return self.lstm_training_started
    
    def redistribute_reward(self, states_var, action, length):
        # Prepare LSTM inputs
        # if not torch.is_tensor(states_var):
        #     states_var = Variable(torch.FloatTensor(states_var)).to(device)
        # Calculate LSTM predictions
        self.model.eval()
        lstm_out, _, attn = self.model(states_var, action, length)
        self.model.train()
        pred_g0 = torch.cat([torch.zeros_like(lstm_out[:, 0:1, :]), lstm_out], dim=1)
        redistributed_reward = pred_g0[:, 1:, 0] - pred_g0[:, :-1, 0]
        # Scale reward back up as LSTM targets have been scaled.
        new_reward = redistributed_reward * self.return_scaling
        return new_reward, attn
    
    def calculate_accuracy(self, y_pred, y):
        top_pred = y_pred.argmax(1, keepdim=True)
        correct = top_pred.eq(y.view_as(top_pred)).sum()
        acc = correct.float() / y.shape[0]
        return acc
    
    # def calculate_loss(self, predicted_G0, returns, length):
        
    #     if not torch.is_tensor(returns):
    #         returns = torch.tensor(returns)
    #         returns = returns.to(device)
    #     prediction = predicted_G0[range(predicted_G0.size(0)), length[:]-1]
    #     trajectory_end_loss = self.loss(prediction, returns)
    #     return trajectory_end_loss
    
    def calculate_main_loss(self, predicted_G0, returns, length):
        return_ = returns.unsqueeze(-1).repeat(1, predicted_G0.size(1))
        if self.config["REWARD_LEARNING"]["is_feedback_binary"]:
            all_timestep_loss = sigmoid_focal_loss(predicted_G0, return_, gamma=5, reduction='none')
        else:
            all_timestep_loss = F.mse_loss(predicted_G0, return_, reduction = 'none')
        len = length[:] - 1
        all_timestep_loss_indexed = all_timestep_loss[range(predicted_G0.size(0)), len]
        return all_timestep_loss_indexed
    
    # def calculate_cnn_loss(self, predicted, target, length):
    #     if self.config["REWARD_LEARNING"]["is_feedback_binary"]:
    #         all_timestep_loss = F.binary_cross_entropy(predicted_G0, return_, reduction = 'none')
    #     else:
    #         all_timestep_loss = F.mse_loss(predicted_G0, return_, reduction = 'none')
    #     # Create the mask
    #     self.mask = torch.zeros_like(all_timestep_loss)
    #     for l_num, l in enumerate(length):
    #         self.mask[l_num, :l] = 1
    #     #Multiplying with the self.mask to avoid the padded sequence
    #     all_timestep_loss = all_timestep_loss * self.mask
    #     # Average for each sequence
    #     self.mean_all_timestep_loss_along_sequence = all_timestep_loss.sum(1) / self.mask.sum(1)
            
    #     # Average across the batch
    #     mean_loss = self.mean_all_timestep_loss_along_sequence.mean()
    #     return mean_loss
    
    def calculate_aux_loss(self, predicted_G0, returns, length):#, pred_q):
        if not torch.is_tensor(returns):
            returns = torch.tensor(returns)
            returns = returns.to(device)
            
        # B x L
        return_ = returns.unsqueeze(-1).repeat(1, predicted_G0.size(1))
        if self.config["REWARD_LEARNING"]["is_feedback_binary"]:
            all_timestep_loss =sigmoid_focal_loss(predicted_G0, return_, gamma=5, reduction='none')
        else:
            all_timestep_loss = F.mse_loss(predicted_G0, return_, reduction = 'none')
            
        # Create the mask
        self.mask = torch.zeros_like(all_timestep_loss)
        for l_num, l in enumerate(length):
            self.mask[l_num, :l] = 1
                
        #Multiplying with the self.mask to avoid the padded sequence
        all_timestep_loss = all_timestep_loss * self.mask

        # Average for each sequence
        self.mean_all_timestep_loss_along_sequence = all_timestep_loss.sum(1) / self.mask.sum(1)
    
        return self.mean_all_timestep_loss_along_sequence 
    
    def q_estimate_loss(self, q_values, q_estimate):
        q_values = q_values[:,3:,...]
        q_values_estimate = q_estimate[:,:-3,...]
        if self.config["REWARD_LEARNING"]["is_feedback_binary"]:
            loss = sigmoid_focal_loss(q_values_estimate, q_values, gamma=5, reduction='mean')
        else:
            loss = F.mse_loss(q_values_estimate, q_values, reduction ='none')
        loss = loss.sum(1).squeeze(-1)
        return loss
        
    # def calculate_loss(self, predicted_G0, returns, length):
        
    #     if not torch.is_tensor(returns):
    #         returns = torch.tensor(returns)
    #         returns = returns.to(device)
            
    #     # B x L
    #     return_ = returns.unsqueeze(-1).repeat(1, predicted_G0.size(1))
    #     all_timestep_loss = F.mse_loss(predicted_G0, return_, reduction = 'none')
            
    #     # Create the mask
    #     self.mask = torch.zeros_like(all_timestep_loss)
    #     for l_num, l in enumerate(length):
    #         self.mask[l_num, :l] = 1
                
    #     #Multiplying with the self.mask to avoid the padded sequence
    #     all_timestep_loss = all_timestep_loss * self.mask

    #     # Average for each sequence
    #     self.mean_all_timestep_loss_along_sequence = all_timestep_loss.sum(1) / self.mask.sum(1)
            
    #     # Average across the batch
    #     mean_loss = self.mean_all_timestep_loss_along_sequence.mean()
    #     aux_loss = self.continuous_pred_factor * mean_loss
        
    #     #all_time_step_loss_np = all_timestep_loss.detach().cpu().numpy()
       
    #     # LSTM is mainly trained on getting the final prediction of g0 right.
    #     len = length[:] - 1
    #     all_timestep_loss_indexed = all_timestep_loss[range(predicted_G0.size(0)), len]
    #     main_loss = all_timestep_loss_indexed.mean()

               
    #     # for params in self.model.lstm_layer.parameters():
    #     #     a = 2 
    
    #     #Calculating reward loss
    #     #reward_loss = self.calculate_reward_loss(predicted_G0, returns, len)
        
    #     #Calculating the aux task loss
    #     #aux_task_loss = self.calculate_aux_task_loss(aux_task_out, predicted_G0)
        
    #     # if torch.isnan(lstm_loss):
    #     #     print("lstm_loss is nan")
        
    #     # LSTM update and loss tracking
        
    #     return main_loss, aux_loss
    
    
    def calculate_aux_task_loss(self, aux_task_out, predicted_G0):
        aux_task_out = aux_task_out.squeeze(-1)
        aux_loss = F.mse_loss(aux_task_out, predicted_G0[:, 10:])
        return self.continuous_pred_factor*aux_loss
    
    def calculate_reward_loss(self, lstm_out, feedback, len):
        #pred_g0 = torch.cat([torch.zeros_like(lstm_out[:, 0:1]), lstm_out], dim=1)
        redistributed_reward = lstm_out[:, 1:] - lstm_out[:, :-1]
        redistributed_reward = redistributed_reward*self.mask[:,1:]
        redistributed_reward = torch.sum(redistributed_reward, dim=1)
        reward_loss = F.mse_loss(redistributed_reward, feedback)
        # log_term = -feedback*torch.log(torch.square(lstm_out[range(lstm_out.size(0)), len]-feedback))
        # mask = log_term!=0.0
        # for params in self.model.lstm_layer.parameters():
        #     a = 2 
        # if torch.isnan(reward_loss):
        #     print("reward loss is nan")
        # if torch.all(mask==False):
        #     return reward_loss
        # else:
        #     y_mean = (log_term*mask).sum(dim=0)/mask.sum(dim=0)
        #     return reward_loss + torch.square(y_mean)
        return reward_loss
    
    def loss(self, x, x_hat, z, z_quantized):
        recon_loss = F.mse_loss(x_hat, x)
        commitment_loss = F.mse_loss(z_quantized.detach(), z)
        vq_loss = self.commitment_cost * commitment_loss
        total_loss = recon_loss + vq_loss
        
        return total_loss
    
    # def vae_loss_function(self, x, x_hat, mean, logvar):
    #     reconstruction_loss = F.mse_loss(x_hat, x, reduction = 'sum')
    #     #reconstruction_loss = F.mse_loss(x_hat, x, size_average=False) / x.size(0) # Reconstruction loss
    #     q_z_x = Normal(mean, logvar.mul(.5).exp())
    #     p_z = Normal(torch.zeros_like(mean), torch.ones_like(logvar))
    #     kl_div = kl_divergence(q_z_x, p_z).sum(1).mean()
    #     total_loss = reconstruction_loss + 0.001*kl_div
    #     #total_loss = reconstruction_loss + self.annealing_factor*kl_div
    #     return total_loss
    
    def vae_loss_function(self, x, recon_x, mu, logvar):
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction = 'sum')
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        #kl_loss = torch.mean(-0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)
        # Total VAE loss
        total_loss = recon_loss + 0.01*kl_loss
        #total_loss = recon_loss + self.annealing_factor*kl_loss
        return total_loss
    
    # def calculate_loss(self, predicted_G0, returns, length):
        """This function calculate loss without mask"""
    #     if not torch.is_tensor(returns):
    #         returns = torch.tensor(returns)
    #         returns = returns.to(device)
        
    #     all_timestep_loss = self.mse_loss(predicted_G0, returns.repeat(1, predicted_G0.size(1)))
    #     aux_loss = self.continuous_pred_factor * all_timestep_loss.mean()
    #     main_loss = all_timestep_loss[range(predicted_G0.size(0)), length[:] - 1].mean()

    #     # LSTM update and loss tracking
    #     lstm_loss = main_loss + aux_loss
        
    #     return lstm_loss, main_loss, aux_loss
    
    def readjust_trajectory_score(self, feedbacks, indices):
        #Need to readjsut the trajectory score so that the sampling probability get updated
        for loss, feedback, index in zip(self.mean_all_timestep_loss_along_sequence, feedbacks, indices):
            trajectory_loss = self.buffer.get_trajectory_score([loss], feedback)
            self.buffer.trajectory_score[index] = trajectory_loss.item()
            self.buffer.lstm_loss[index] = loss.detach().cpu().numpy()
    
    def extract_subtrajectory(self, states, action, mask):
        state_array = np.zeros(shape=(states.size(0), states.size(1), states.size(2), states.size(3), states.size(4)), dtype=np.float32)
        action_array = np.zeros(shape=(action.size(0), action.size(1)), dtype=np.float32)
        for i in range(states.size(0)):
            length = np.sum(mask[i,:])
            extracted_state = states.detach().cpu().numpy()[i, mask[i,:], :,:,:]
            extracted_action = action.detach().cpu().numpy()[i, mask[i,:]]
            state_array[i, :length] = extracted_state
            action_array[i, :length] = extracted_action
        return torch.Tensor(state_array).to(device), torch.Tensor(action_array).to(device)
    
    def adjust_lstm_weight(self, num_units):
        self.model.lstm_layer.lstm_layer.bias_hh_l0.data[num_units:2*num_units-1].fill_(1e5)
        self.model.lstm_layer.lstm_layer.bias_hh_l1.data[num_units:2*num_units-1].fill_(1e5)
        self.model.lstm_layer.lstm_layer.bias_hh_l0.data[3*num_units:-1].fill_(1e5)
        self.model.lstm_layer.lstm_layer.bias_hh_l1.data[3*num_units:-1].fill_(1e5)
        
    def create_data_for_heatmap(self, redistributed_reward, obs):
        obs = obs.detach().cpu().numpy()
        obs = obs.astype(int)
        data = np.zeros((9,9))
        for count, reward in enumerate(redistributed_reward):
            data[obs[count][1], obs[count][0]] = max(data[obs[count][1], obs[count][0]], reward)
        return data
    
    def update_model(self, loss, model, optimizer, retain_graph=False):
        optimizer.zero_grad()
        loss.backward(retain_graph = retain_graph)
        clip_grad_norm_(model.parameters(), 0.50)
        optimizer.step()
        
             
    # Trains the LSTM until -on average- the main loss is below 0.25.
    def train(self):
        if self.config["REWARD_LEARNING"]['is_load_lstm']:
            self.model.load_state_dict(torch.load(self.config["REWARD_LEARNING"]['model_dir'])['lstm_weight'])
            print("[INFO] LSTM model loaded from ", self.config["REWARD_LEARNING"]['model_dir'])
        else:
            lstm_update = 0
            lstm_n_updates = 50000#self.config["REWARD_LEARNING"]["n_update"]
            pbar_lstm = tqdm(total=lstm_n_updates)
            #Uniform sampling of data from the collected trajectory data of agent to train the LSTM
            # Create a subset of main lesson buffer to train the LSTM. This should be same as that of the PEBBLE training
            train_observations, train_action, rewards, train_len, indices = self.buffer.sample(self.config["REWARD_LEARNING"]["size"])
            train_observations = torch.tensor(train_observations).to(device)
            train_action = torch.tensor(train_action).to(device)
            rewards = torch.tensor(rewards).to(device)
            train_len = torch.tensor(train_len).to(device)
            state_encoding = train_observations
            while lstm_update < lstm_n_updates:
                # Get samples from the lesson buffer and prepare them.
                # Create a subset of main lesson buffer to train the LSTM. This should be same as that of the PEBBLE training
                #train_observations, train_action, rewards, train_len, indices = self.buffer.sample(self.config["REWARD_LEARNING"]["batch_size"])
                returns = torch.sum(rewards, 1, keepdim=True)
                q_values, q_estimate, _ = self.model(state_encoding, train_action, train_len)
                #Calculating loss
                main_loss = self.calculate_main_loss(q_values.squeeze(-1), returns.squeeze(-1), train_len.type(torch.int64).squeeze(-1))
                aux_loss = self.calculate_aux_loss(q_values.squeeze(-1), returns.squeeze(-1), train_len.type(torch.int64).squeeze(-1))#, rudder_out_pred)     
                q_estimate_loss = self.q_estimate_loss(q_values, q_estimate)
                trajectory_end_loss = main_loss.mean() + self.continuous_pred_factor*(aux_loss.mean() + q_estimate_loss.mean())
                self.update_model(trajectory_end_loss, self.model, self.lstm_optimizer, False)
                main_loss = trajectory_end_loss.item()
                #Updating the priorities
                total_loss_trajectory_wise = main_loss + self.continuous_pred_factor*(q_estimate_loss + trajectory_end_loss)
                total_loss_trajectory_wise = total_loss_trajectory_wise.detach().cpu().numpy()
                self.buffer.do_post_update_works(indices, total_loss_trajectory_wise, lstm_update)
                log_value = [lstm_update, main_loss]
                write_log(self.logger, log_keys, log_value, None)
                pbar_lstm.set_description("LSTM Loss {}".format(main_loss))
                pbar_lstm.update(1)
                #Dumping the LSTM model
                if lstm_update % 15000 == 0:
                    checkpoint = {"lstm_weight" : self.model.state_dict()}
                    path = self.dump_dir + '/lstm_{}.tar'.format(lstm_update)
                    torch.save(checkpoint, path)
                lstm_update += 1
            pbar_lstm.close()
