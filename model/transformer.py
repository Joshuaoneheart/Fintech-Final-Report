import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Seq_Encode(nn.Module):
  def __init__(self, config,input_dim = 234, output_dim = 768, dropout=0.1, device = "cuda:0"):
    super().__init__()
    # Project the dimension of features from that of input into d_model.
    # TODO:
    #   Change Transformer to Conformer.
    #   https://arxiv.org/abs/2005.08100
    self.prenet = nn.Linear(input_dim, output_dim)
    self.config = config
    self.encoderlayer = nn.TransformerEncoderLayer(d_model = output_dim, dim_feedforward = 1024, nhead=2) 
    self.decoderlayer = nn.TransformerDecoderLayer(d_model = output_dim, dim_feedforward = 1024, nhead=2) 

    self.encoder = nn.TransformerEncoder(self.encoderlayer, num_layers = 3)
    self.decoder = nn.TransformerDecoder(self.decoderlayer, num_layers = 3)

  def forward(self, batch, labels, steps, device, src_padding_mask, tgt_padding_mask):
    """
    args:
      mels: (batch size, length, 40)
    return:
      out: (batch size, n_spks)
    """
    encoder_out = self.encoder(self.prenet(batch).permute(1, 0, 2), src_key_padding_mask = src_padding_mask)
    tgtmask = (torch.triu(torch.ones(self.config["max_embedding_len"], self.config["max_embedding_len"])) == 1).transpose(0, 1)
    tgtmask = tgtmask.float().masked_fill(tgtmask == 0, float("-inf")).masked_fill(tgtmask == 1, float(0.0)).to(device)
    start_ = torch.zeros(1, 1, 768).to(device)
    outputs = self.decoder(torch.cat([start_, labels.permute(1, 0, 2)[:-1,:,:]], dim = 0) , memory = encoder_out, tgt_mask = tgtmask, tgt_key_padding_mask = tgt_padding_mask)
    schedule = (torch.rand(labels.shape[1], labels.shape[0]) >= min((steps / 400000) + 0.2, 0.7))
    outputs = torch.cat([start_, outputs],dim = 0)[:-1, :, :]
    for idx, batch in enumerate(outputs):
        for idx_ in range(len(batch)):
            if schedule[idx, idx_].item() and idx + 1 < self.config["max_embedding_len"]:
                outputs[idx + 1,idx_,:] = labels[idx_, idx, :]
    outputs = self.decoder(outputs.detach() , memory = encoder_out, tgt_mask = tgtmask, tgt_key_padding_mask = tgt_padding_mask)
    return outputs

  def generate(self, batch, labels, length, device, src_padding_mask, tgt_padding_mask):
    """
    args:
      mels: (batch size, length, 40)
    return:
      out: (batch size, n_spks)
    """
    encoder_out = self.encoder(self.prenet(batch).permute(1, 0, 2), src_key_padding_mask = src_padding_mask)
    start_ = torch.zeros(1, 1, 768).to(device)
    outputs = torch.zeros(length[0] + 1, 1, 768).float().to(device)
    outputs[0, :, :] = start_
    for i in range(1, length[0] + 1):
        tgtmask = (torch.triu(torch.ones(i, i)) == 1).transpose(0, 1)
        tgtmask = tgtmask.float().masked_fill(tgtmask == 0, float("-inf")).masked_fill(tgtmask == 1, float(0.0)).to(device)
        tmp_out = self.decoder(outputs[:i, :, :] , memory = encoder_out, tgt_mask = tgtmask, tgt_key_padding_mask = tgt_padding_mask[:,:i])
        outputs[i, :, :] = tmp_out[-1, :, :]
        
    return outputs[1:, :, :]
