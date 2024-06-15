
import torch.nn as nn

class FiLM(nn.Module):
    def __init__(self, input_size, conditioning_size):
        super(FiLM, self).__init__()
        self.conditioning_net = nn.Sequential(
            nn.Linear(conditioning_size, input_size),
            nn.ReLU()
        )
        self.gamma = nn.Linear(input_size, input_size)
        self.beta = nn.Linear(input_size, input_size)
    
    def forward(self, input, conditioning):
        gamma = self.gamma(self.conditioning_net(conditioning))
        beta = self.beta(self.conditioning_net(conditioning))
        modulated_input = input * gamma + beta
        return modulated_input