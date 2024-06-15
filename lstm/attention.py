import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class TransformerEncoderCustom(nn.TransformerEncoderLayer):
  # modified to return attention weights in addition to output
  def forward(self, src, src_mask = None, src_key_padding_mask = None, output_attention=False):
      """Pass the input through the encoder layer.

      Args:
          src: the sequence to the encoder layer (required).
          src_mask: the mask for the src sequence (optional).
          src_key_padding_mask: the mask for the src keys per batch (optional).

      Shape:
          see the docs in Transformer class.
      """
      src2, attention = self.self_attn(src, src, src, attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask, need_weights=output_attention)
      attention = attention.detach().clone()
      src = src + self.dropout1(src2)
      src = self.norm1(src)
      src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
      src = src + self.dropout2(src2)
      src = self.norm2(src)

      return src, attention


class SelfAttentionForRL(nn.Module):
  def __init__(
      self,
      observation_size,
      action_size,
      device,
      embedding_size=32, # chosen odd so that embedding_size+action_size is even
      dim_feedforward=32,
      pad_val=10,
      max_len=250,
      verbose=False,
      ):
    super(SelfAttentionForRL, self).__init__()

    self.verbose = verbose
    self.device = device
    self.pad_val = pad_val
    # self.linear1 = nn.Sequential( # TODO: add dropout?
    #     nn.Linear(observation_size, embedding_size, bias=True),
    #     nn.ReLU()
    # )
    # self.linear2 = nn.Sequential( # TODO: add dropout?
    #     nn.Linear(embedding_size + action_size, embedding_size + action_size),
    #     nn.ReLU()
    # )
    self.self_attention = TransformerEncoderCustom(observation_size + action_size, nhead=4, dim_feedforward=dim_feedforward, dropout=0.2)
    #self.fc_out = nn.Linear(embedding_size + action_size, 1, bias=True) 
    
  def forward(self, observations, train_len, output_attention=False):
    #observations_np = observations.detach().cpu().numpy()
    observations = torch.permute(observations, (1,0,2))
    
    #TODO Slice the observation according to the actual length of the observation, I think padding mask can do the job
    
    seq_length, N, observation_size = observations.shape
    #x = self.linear1(observations) 
    
    src_mask = self.generate_square_subsequent_mask(sz=seq_length)
    padding_mask = self.make_padding_mask(observations, train_len)

    out, attention = self.self_attention(
        observations, 
        src_mask=src_mask, 
        src_key_padding_mask=padding_mask,
        output_attention=True
    )
    #out = self.fc_out(out)
    out = torch.permute(out, (1,0,2))
    if output_attention:
      return (out, attention)
    else:
      return out

  # From pytorch Transformer source
  def generate_square_subsequent_mask(self, sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask.to(self.device)

  def make_padding_mask(self, batch_of_sequences, train_len):
    """This function creates a mask to avoid attention to padded values."""
    # batch_of_sequences (S, N, E)
    # return (N, S)
    
    #TODO Change the function lines to suit to the scenario
    padding_mask = torch.zeros((batch_of_sequences.size(1), batch_of_sequences.size(0)), dtype=torch.int64)
    for count, len in enumerate(train_len.squeeze(-1)):
      padding_mask[count, len.type(torch.int64):] = 1

    return padding_mask.to(self.device) > 0


class Attention(nn.Module):
  def __init__(self, query_dim, key_dim, value_dim):
    super(Attention, self).__init__()
    self.scale = 1. / math.sqrt(query_dim)

  def forward(self, query, keys, values):
    query = query.repeat(keys.size(1), 1, 1)
    # Query = [TxBxQ]
    # Keys = [TxBxK]
    keys = torch.permute(keys, (1,0,2))
    # Values = [TxBxV]
    values = torch.permute(values, (1,0,2))
    # Outputs = a:[TxB], lin_comb:[BxV]

    # Here we assume q_dim == k_dim (dot product attention)

    query = torch.permute(query, (1,0,2)) # [TxBxQ] -> [BxTxQ]
    keys = keys.transpose(0,1).transpose(1,2) # [TxBxK] -> [BxKxT]
    energy = torch.bmm(query, keys) # [Bx1xQ]x[BxKxT] -> [Bx1xT]
    energy = F.softmax(energy.mul_(self.scale), dim=2) # scale, normalize

    values = values.transpose(0,1) # [TxBxV] -> [BxTxV]
    linear_combination = torch.bmm(energy, values).squeeze(1) #[Bx1xT]x[BxTxV] -> [BxV]
    return energy, linear_combination