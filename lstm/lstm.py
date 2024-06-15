import torch
import torch.nn as nn
from widis_lstm_tools.nn import LSTMLayer

class LSTM(nn.Module):
    def __init__(self, fc_input_size, n_units):
        super(LSTM, self).__init__()
        self.lstm_layer1 = LSTMLayer(in_features=fc_input_size, out_features=n_units,
                              w_ci=(lambda *args, **kwargs: nn.init.normal_(mean=0, std=0.1, *args, **kwargs), False),
                              w_ig=(False, lambda *args, **kwargs: nn.init.normal_(mean=0, std=0.1, *args, **kwargs)),
                              w_og=False,
                              b_ci=lambda *args, **kwargs: nn.init.normal_(mean=0, *args, **kwargs),
                              b_ig=lambda *args, **kwargs: nn.init.normal_(mean=-5, *args, **kwargs),
                              b_og=False,
                              a_out=lambda x: x
                              )
        # self.lstm_layer2 = LSTMLayer(in_features=fc_input_size, out_features=n_units,
        #                       w_ci=(lambda *args, **kwargs: nn.init.normal_(mean=0, std=0.1, *args, **kwargs), False),
        #                       w_ig=(False, lambda *args, **kwargs: nn.init.normal_(mean=0, std=0.1, *args, **kwargs)),
        #                       w_og=False,
        #                       b_ci=lambda *args, **kwargs: nn.init.normal_(mean=0, *args, **kwargs),
        #                       b_ig=lambda *args, **kwargs: nn.init.normal_(mean=-5, *args, **kwargs),
        #                       b_og=False,
        #                       a_out=lambda x: x
        #                       )
        #self.lstm_layer = nn.LSTM(input_size = fc_input_size, hidden_size = n_units, batch_first = True, num_layers=1, bidirectional = True)
    def forward(self, x):
        #x_np = x.detach().cpu().numpy()
        forward_x = self.lstm_layer1(x, return_all_seq_pos = True)
        #flipped_x = torch.flip(x, [1])
        #flipped_x_np = flipped_x.detach().cpu().numpy()
        #backward_x = self.lstm_layer2(flipped_x, return_all_seq_pos = True)
        #final_out = torch.cat([forward_x[0], backward_x[0]], dim = -1)
        #x = self.lstm_layer2(x[0], return_all_seq_pos = True)
        #x = self.lstm_layer(x)
        return forward_x[0]