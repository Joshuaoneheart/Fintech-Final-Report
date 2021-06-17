import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Seq_Encode(nn.Module):
  def __init__(self, input_dim = 234, output_dim = 768, dropout=0.1, device = "cuda"):
    super().__init__()
    # Project the dimension of features from that of input into d_model.
    # TODO:
    #   Change Transformer to Conformer.
    #   https://arxiv.org/abs/2005.08100
    self.prenet = nn.Linear(input_dim, output_dim)
    self.decoderlayer = nn.TransformerDecoderLayer(d_model = output_dim, dim_feedforward = 256, nhead=2) 

    self.encoder = ConformerEncoder(input_dim = output_dim, encoder_dim = 128, num_layers = 2, num_attention_heads=2, device = device)
    self.enc_2_dec = nn.Linear(128, 768)
    self.decoder = nn.TransformerDecoder(self.decoderlayer, num_layers = 3)

  def forward(self, batch, labels, steps, device):
    out = self.prenet(batch)
    encoder_out, _ = self.encoder(out, out.size(1))
    encoder_out = self.enc_2_dec(encoder_out)
    tgtmask = (torch.triu(torch.ones(my_config["max_embedding_len"], my_config["max_embedding_len"])) == 1).transpose(0, 1)
    tgtmask = tgtmask.float().masked_fill(tgtmask == 0, float("-inf")).masked_fill(tgtmask == 1, float(0.0)).to(device)
    outputs = self.decoder(labels.permute(1, 0, 2) , memory = encoder_out, tgt_mask = tgtmask)
    schedule = (torch.rand(labels.shape[1], labels.shape[0]) >= steps / 200000)
    for idx, batch in enumerate(outputs):
        for idx_ in range(len(batch)):
            if schedule[idx, idx_].item():
                outputs[idx,idx_,:] = labels[idx_, idx, :]
    outputs = self.decoder(outputs , memory = encoder_out, tgt_mask = tgtmask)
    return outputs

  def generate(self, batch, labels, length, device):
    """
    args:
      mels: (batch size, length, 40)
    return:
      out: (batch size, n_spks)
    """
    out = self.prenet(batch)
    encoder_out, _ = self.encoder(out, out.size(1))
    encoder_out = self.enc_2_dec(encoder_out)
    start_ = labels[:, 0, :]
    outputs = torch.zeros(length[0], 1, 768).float().to(device)
    outputs[0, :, :] = start_.unsqueeze(0).permute(1, 0, 2)
    for i in range(1, length[0]):
        tgtmask = (torch.triu(torch.ones(i, i)) == 1).transpose(0, 1)
        tgtmask = tgtmask.float().masked_fill(tgtmask == 0, float("-inf")).masked_fill(tgtmask == 1, float(0.0)).to(device)
        tmp_out = self.decoder(outputs[:i, :, :] , memory = encoder_out, tgt_mask = tgtmask)
        outputs[i, :, :] = tmp_out[-1, :, :]
        
    return outputs
