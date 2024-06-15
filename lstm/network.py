import numpy as np
import torch
import torch.nn as nn
from utils.utils import get_device, to_one_hot, custom_action_encoding
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Network(nn.Module):
    def __init__(self, linear_layer, lstm_layer, attention_layer, pre_linear_layer):#, conv_layer):#, embedding_layer):#, aux_layer):
        super(Network, self).__init__()

        self.pre_linear_layer = pre_linear_layer
        #self.cnn_layer = conv_layer
        self.lstm_layer = lstm_layer
        self.linear_layer = linear_layer
        #self.aux_linear = aux_layer
        #self.embedding = embedding_layer
        self.attn_layer = attention_layer
        
        
    # def get_cnn_features(self, data):
    #     if len(data.size()) != 5:
    #         data = data.unsqueeze(0)
    #     b,t,c,h,w = data.size()
    #     data = data.view(b*t,c,h,w)
    #     cnn_output = self.cnn_layer(data)
    #     return cnn_output.view(b,t,-1)
        
    

    def forward(self, states, action, length):
        states = self.pre_linear_layer(states)
        #attn_out = self.attn_model(states, length)
        #b,t,f = states.size()
        #states = states.view(b, t, 5, 5)
        #states = states.view(b*t, 5, 5)
        
        #state_features = cnn_output.view(b, t, 9)
        #cnn_output = self.get_cnn_features(states)
        action_embed = torch.tensor(action.clone().detach()).to(device)
        #action_one_hot = torch.tensor(custom_action_encoding(action, 24)).type(torch.float32).to(device)
        lstm_input = torch.cat((states, action_embed), dim = -1)
        lstm_output = self.lstm_layer(lstm_input)
        #lstm_output, unpacked_len = torch.nn.utils.rnn.pad_packed_sequence(lstm_output[0], batch_first =True)
        #energy, linear_combination = self.attn_model(lstm_output[1]['c'][-1], lstm_output[0], lstm_output[0])
        attention_weights = F.softmax(self.attn_layer(lstm_output[0]), dim=1)
        attended_out = attention_weights * lstm_output[0]
        q_value = self.linear_layer(attended_out)
        #q_value_preiction = self.aux_linear(lstm_output[0])
        #aux_task_input = lstm_output[:, :-10, :]
        #aux_task_output = self.aux_linear(aux_task_input)
        return q_value#, q_value_preiction