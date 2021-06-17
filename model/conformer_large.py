import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from conformer.encoder import ConformerEncoder


class Seq_Encode(nn.Module):
  def __init__(self, config, input_dim = 234, output_dim = 768, dropout=0.1, device = "cuda"):
    super().__init__()
    # Project the dimension of features from that of input into d_model.
    # TODO:
    #   Change Transformer to Conformer.
    #   https://arxiv.org/abs/2005.08100
    self.prenet = nn.Linear(input_dim, output_dim)
    self.config = config
    self.decoderlayer = nn.TransformerDecoderLayer(d_model = output_dim, dim_feedforward = 256, nhead=2) 

    self.encoder = ConformerEncoder(input_dim = output_dim, encoder_dim = 512, num_layers = 2, num_attention_heads=2, device = device)
    self.enc_2_dec = nn.Linear(512, 768)
    self.decoder = nn.TransformerDecoder(self.decoderlayer, num_layers = 3)

  def forward(self, batch, labels, steps, device, src_padding_mask, tgt_padding_mask):
    out = self.prenet(batch)
    encoder_out, _ = self.encoder(out, out.size(1))
    encoder_out = self.enc_2_dec(encoder_out)
    tgtmask = (torch.triu(torch.ones(self.config["max_embedding_len"], self.config["max_embedding_len"])) == 1).transpose(0, 1)
    tgtmask = tgtmask.float().masked_fill(tgtmask == 0, float("-inf")).masked_fill(tgtmask == 1, float(0.0)).to(device)
    outputs = self.decoder(labels.permute(1, 0, 2) , memory = encoder_out, tgt_mask = tgtmask, tgt_key_padding_mask = tgt_padding_mask)
    schedule = (torch.rand(labels.shape[1], labels.shape[0]) >= steps / 5000)
    i = 0
    for idx, batch in enumerate(outputs):
        for idx_ in range(len(batch)):
            if schedule[idx, idx_].item() or idx == 0:
                outputs[idx,idx_,:] = labels[idx_, idx, :]
                i += 1
    outputs = self.decoder(outputs.detach() , memory = encoder_out, tgt_mask = tgtmask, tgt_key_padding_mask = tgt_padding_mask)
    return outputs

  def generate(self, batch, labels, length, device, src_padding_mask, tgt_padding_mask):
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
        tmp_out = self.decoder(outputs[:i, :, :] , memory = encoder_out, tgt_mask = tgtmask, tgt_key_padding_mask = tgt_padding_mask[:, :i])
        outputs[i, :, :] = tmp_out[-1, :, :]
        
    return outputs


"""# Learning rate schedule
- For transformer architecture, the design of learning rate schedule is different from that of CNN.
- Previous works show that the warmup of learning rate is useful for training models with transformer architectures.
- The warmup schedule
  - Set learning rate to 0 in the beginning.
  - The learning rate increases linearly from 0 to initial learning rate during warmup period.
"""

import math

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


def get_cosine_schedule_with_warmup(
  optimizer: Optimizer,
  num_warmup_steps: int,
  num_training_steps: int,
  num_cycles: float = 0.5,
  last_epoch: int = -1,
):
  """
  Create a schedule with a learning rate that decreases following the values of the cosine function between the
  initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
  initial lr set in the optimizer.

  Args:
    optimizer (:class:`~torch.optim.Optimizer`):
      The optimizer for which to schedule the learning rate.
    num_warmup_steps (:obj:`int`):
      The number of steps for the warmup phase.
    num_training_steps (:obj:`int`):
      The total number of training steps.
    num_cycles (:obj:`float`, `optional`, defaults to 0.5):
      The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
      following a half-cosine).
    last_epoch (:obj:`int`, `optional`, defaults to -1):
      The index of the last epoch when resuming training.

  Return:
    :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
  """

  def lr_lambda(current_step):
    # Warmup
    if current_step < num_warmup_steps:
      return float(current_step) / float(max(1, num_warmup_steps))
    # decadence
    progress = float(current_step - num_warmup_steps) / float(
      max(1, num_training_steps - num_warmup_steps)
    )
    return max(
      0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))
    )

  return LambdaLR(optimizer, lr_lambda, last_epoch)

"""# Model Function
- Model forward function.
"""

import torch


def model_fn(batch, model, criterion, device, step):
  """Forward a batch through the model."""

  mels, labels, lengths = batch
  mels = mels.to(device)
  labels = labels.to(device)

  outs = model(mels, labels, step, device)
  loss = 0
  for idx, label in enumerate(labels):
      y = torch.ones(1, lengths[idx]).to(device)
      loss = criterion(outs.permute(1, 0, 2)[idx,:lengths[idx],:].contiguous().view(-1, 768), label[:lengths[idx], :].contiguous().view(-1, 768), y.view(-1))
  return loss

"""# Validate
- Calculate accuracy of the validation set.
"""

from tqdm import tqdm
import torch


def valid(dataloader, model, criterion, device): 
  """Validate on validation set."""

  model.eval()
  running_loss = 0.0
  pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")
  for i, batch in enumerate(dataloader):
    with torch.no_grad():
      loss = model_fn(batch, model, criterion, device, i)
      running_loss += loss.item()

    pbar.update(dataloader.batch_size)
    pbar.set_postfix(
      loss=f"{running_loss / (i+1)}",
    )

  pbar.close()
  model.train()
  return running_loss / len(dataloader)

"""# Main function"""

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, random_split


def parse_args():
  """arguments"""
  config = {
    "data_dir": "./Dataset",
    "save_path": "conformer.ckpt",
    "batch_size": 1,
    "n_workers": 0,
    "valid_steps": 800000,
    "warmup_steps": 1000,
    "save_steps": 10000,
    "total_steps": 800000,
  }

  return config


def main(
  data_dir,
  save_path,
  batch_size,
  n_workers,
  valid_steps,
  warmup_steps,
  total_steps,
  save_steps,
):
  """Main function."""
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  print(f"[Info]: Use {device} now!")

  train_loader, valid_loader = get_dataloader(data_dir, batch_size, n_workers)
  train_iterator = iter(train_loader)
  print(f"[Info]: Finish loading data!",flush = True)

  model = Seq_Encode(device = device).to(device)
  model.load_state_dict(torch.load(save_path))
  criterion = nn.CosineEmbeddingLoss()
  optimizer = AdamW(model.parameters(), lr=1e-5)
  scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
  print(f"[Info]: Finish creating model!",flush = True)

  best_state_dict = None

  pbar = tqdm(total=valid_steps / 100, ncols=0, desc="Train", unit=" step")
  best_loss = 10
  losses = 0
  for step in range(total_steps):
    # Get data
    try:
      batch = next(train_iterator)
    except StopIteration:
      train_iterator = iter(train_loader)
      batch = next(train_iterator)

    loss = model_fn(batch, model, criterion, device, step)
    batch_loss = loss.item()
    losses += batch_loss

    # Updata model
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()
    
    # Log
    if (step+1) % 100 == 0:
        pbar.update()
        pbar.set_postfix(
          loss=f"{losses / 100:.6f}",
          step=step + 1,
        )
        losses = 0

    # Do validation
    if (step + 1) % valid_steps == 0:
      pbar.close()

      # valid_loss = valid(valid_loader, model, criterion, device)
      valid_loss = 1

      # keep the best model
      if valid_loss <= best_loss:
        best_loss = valid_loss
        best_state_dict = model.state_dict()

      pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    # Save the best model so far.
    if (step + 1) % save_steps == 0:# and best_state_dict is not None:
      torch.save(model.state_dict(), save_path + f"_{step}")
      pbar.write(f"Step {step + 1}, best model saved. (loss={1 - best_loss:.4f})")

  pbar.close()

if __name__ == "__main__":
  main(**parse_args())



"""# Inference

## Dataset of inference
"""

import os
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset

"""## Main funcrion of Inference"""

import json
import csv
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

def parse_args():
  """arguments"""
  config = {
    "data_dir": "./Dataset",
    "model_path": "./conformer.ckpt",
    "output_path": "./conformer.csv",
  }

  return config


def main(
  data_dir,
  model_path,
  output_path,
):
  """Main function."""
  device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
  print(f"[Info]: Use {device} now!")

  dataset = myDataset(data_dir)
  dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=8,
    collate_fn=collate_batch,
  )
  print(f"[Info]: Finish loading data!",flush = True)

  model.load_state_dict(torch.load("model.ckpt"))
  model.eval()
  print(f"[Info]: Finish creating model!",flush = True)

  results = [["Id", "Category"]]
  for mels, labels, lengths in tqdm(dataloader):
    with torch.no_grad():
      mels = mels.to(device)
      labels = labels.to(device)
      outs = model.generate(mels, labels, lengths)


if __name__ == "__main__":
  main(**parse_args())
