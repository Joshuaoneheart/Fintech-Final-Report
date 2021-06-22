#!/usr/bin/python3.9

# -*- coding: utf-8 -*-
"""FinalProject

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1sruNbvRK1w6h0fYgRHKDZ5Vh3pLOjxnb
"""


"""# Setup PyTroch Property

"""

import torch
import numpy as np
# fix random seed
def za_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  
    np.random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

za_seed(1126)

"""# Config"""
from config import my_config 

"""# Data

## Dataset
- Original dataset is [MATBN 中文廣播新聞語料庫]
"""

import os
import tqdm
import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
DATA_DIR = "./"
from utils.data import collate_batch, myDataset
"""## Dataloader
- Split dataset into training dataset(90%) and validation dataset(10%).
- Create dataloader to iterate the data.

"""

import torch
from torch.utils.data import DataLoader, random_split

def get_dataloader(data_dir, batch_size, n_workers):
  """Generate dataloader"""
  dataset = myDataset(my_config, "train")
  # Split dataset into training dataset and validation dataset
  trainlen = int(0.9 * len(dataset))
  lengths = [trainlen, len(dataset) - trainlen]
  trainset, validset = random_split(dataset, lengths)

  train_loader = DataLoader(
    trainset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=n_workers,
    pin_memory=True,
    collate_fn=collate_batch,
  )
  valid_loader = DataLoader(
    validset,
    batch_size=batch_size,
    num_workers=n_workers,
    drop_last=True,
    pin_memory=True,
    collate_fn=collate_batch,
  )

  return train_loader, valid_loader

"""# Model
- TransformerEncoderLayer:
  - Base transformer encoder layer in [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
  - Parameters:
    - d_model: the number of expected features of the input (required).

    - nhead: the number of heads of the multiheadattention models (required).

    - dim_feedforward: the dimension of the feedforward network model (default=2048).

    - dropout: the dropout value (default=0.1).

    - activation: the activation function of intermediate layer, relu or gelu (default=relu).

- TransformerEncoder:
  - TransformerEncoder is a stack of N transformer encoder layers
  - Parameters:
    - encoder_layer: an instance of the TransformerEncoderLayer() class (required).

    - num_layers: the number of sub-encoder-layers in the encoder (required).

    - norm: the layer normalization component (optional).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from model.transformer import Seq_Encode

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

  mels, labels, lengths, src_padding_mask, tgt_padding_mask = batch
  mels = mels.to(device)
  src_padding_mask = src_padding_mask.to(device).bool()
  tgt_padding_mask = tgt_padding_mask.to(device).bool()
  labels = labels.to(device)

  outs = model(mels, labels, step, device, src_padding_mask, tgt_padding_mask)
  loss = 0
  for idx, label in enumerate(labels):
      y = torch.ones(1, lengths[idx]).to(device)
      loss += criterion(outs.permute(1, 0, 2)[idx,:lengths[idx],:].contiguous().view(-1, 768), label[:lengths[idx], :].contiguous().view(-1, 768), y.view(-1))
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
    "batch_size": 1,
    "n_workers": 0,
    "valid_steps": 800000,
    "warmup_steps": 1000,
    "save_steps": 2000,
    "total_steps": 800000,
  }

  return config


def main(
  data_dir,
  batch_size,
  n_workers,
  valid_steps,
  warmup_steps,
  total_steps,
  save_steps,
):
  """Main function."""
  device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
  print(f"[Info]: Use {device} now!")

  train_loader, valid_loader = get_dataloader(data_dir, batch_size, n_workers)
  train_iterator = iter(train_loader)
  print(f"[Info]: Finish loading data!",flush = True)

  model = Seq_Encode(my_config, device = device).to(device)
  model.load_state_dict(torch.load(my_config["ckpt_name"]))
  criterion = nn.CosineEmbeddingLoss()
  optimizer = AdamW(model.parameters(), lr=1e-5)
  scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
  print(f"[Info]: Finish creating model!",flush = True)

  pbar = tqdm(total=total_steps / 100, ncols=0, desc="Train", unit=" step")
  best_loss = 10
  losses = 0
  accu_step = my_config["accu_step"]
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
    loss /= accu_step
    # Updata model
    loss.backward()
    # gradient accumulation
    if ((step + 1) % accu_step) == 0:
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

    # Save the best model so far.
    if (step + 1) % save_steps == 0:
      torch.save(model.state_dict(), my_config["ckpt_name"])
      pbar.write(f"Step {step + 1}, best model saved. (loss={1 - best_loss:.4f})")

  pbar.close()

if __name__ == "__main__":
  main(**parse_args())
