import torch
import torch.nn as nn
from utils.utils import custom_action_encoding
from learning_agent.architectures.mlp import MLP
from lstm.film import FiLM


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = "cpu"

class PEBBLE(nn.Module):
    def __init__(self,
                 config,
                 input_size = 25,
                 output_size = 1,
                 hidden_sizes = [256,512]):
        super(PEBBLE, self).__init__()
        self.model = MLP(input_size = input_size,
                         output_size = output_size,
                         hidden_sizes = hidden_sizes,
                         output_activation = torch.tanh).to(device)
        self.config = config
        self.n_actions = self.config["REWARD_LEARNING"]["n_actions"]
        self.action_embedding_dim = self.config["REWARD_LEARNING"]["action_embedding_dim"]
        self.film = FiLM(self.config["REWARD_LEARNING"]["action_embedding_dim"], self.config["REWARD_LEARNING"]["feature_size"]).to(device)
        
    def forward(self, states, action, len):
        action_one_hot = custom_action_encoding(action, self.n_actions, self.action_embedding_dim)
        action_one_hot = torch.tensor(action_one_hot).to(device)
        if self.config["REWARD_LEARNING"]["is_FiLM"]:
            input = self.film(action_one_hot, states)
        else:
            input = torch.cat((states, action_one_hot), dim=-1)
        return self.model(input), None, None