"""# Data

## Dataset
- Original dataset is [MATBN 中文廣播新聞語料庫]
"""

import os
from tqdm import tqdm
import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from config import my_config
 
 
class myDataset(Dataset):
  def __init__(self, config, split="test"):
    # data: [batch, label]
    if split != "test":
        self.phone_dir = config["phone_dir"]
        self.embedding_dir = config["embedding_dir"]
    else:
        self.phone_dir = config["test_phone_dir"]
        self.embedding_dir = config["test_embedding_dir"]

    phone_files = os.listdir(self.phone_dir)
    self.data = []
    self.names = []
    self.split = split
    size = 0
    for idx, phones in enumerate(tqdm(phone_files)):
      name = phones.replace(".json", "")
      self.names.append(name)
    #   if size >= 4000:
    #       break
      if name not in os.listdir(self.embedding_dir):
        print(f"{name} in phone dir but not in embedding dir")
        continue
      embedding_files = os.listdir(self.embedding_dir + "/" + name)
      has_place = []
      for embeddings in embedding_files:
        _, start, end, _ = embeddings.split(".")
        has_place.append((start, end))
      has_place.sort()
      record_s, record_e = 0, 0
      i = 0
      while i < len(has_place):
        start, end = has_place[i]
        if start != record_s and record_e != 0: 
            size += 1
            self.data.append((idx, int(record_s), int(record_e) + 1))
            while i < len(has_place) and has_place[i][1] <= record_e:
                i += 1
            
        record_s, record_e = start, end
        i += 1
      size += 1
      self.data.append((idx, int(start), int(end)+1))
    print(size)

      
    
  def __len__(self):
    return len(self.data)
 
  def __getitem__(self, index):
    while True:
        try:
            idx, s, e = self.data[index]
            name = self.names[idx]
            with open(self.phone_dir + "/" + name + ".json", "r") as fp:
                ph = json.load(fp)
            with open(self.embedding_dir + "/" + name + "/" + name + "." + str(s) + "." + str(e - 1) + ".json", "r") as fp:
                ems = json.load(fp)
            length = len(ems)
            probs = torch.FloatTensor(ph[s: e])
            embeddings = torch.FloatTensor(ems)
            return probs, embeddings, length
        except:
            idx, s, e = self.data[index]
            print(self.data[index])
            print(self.embedding_dir + "/" + name + "/" + name + "." + str(s) + "." + str(e - 1) + ".json")
            index += 1
import torch
from torch.utils.data import DataLoader, random_split


def collate_batch(batch):
  # Process features within a batch.
  """Collate a batch of data."""
  probs, embeddings, lengths = zip(*batch)
  src_padding_mask = torch.zeros((len(probs), max(probs[0].shape[0],my_config["max_prob_len"])))
  tgt_padding_mask = torch.zeros((len(probs), my_config["max_embedding_len"]))
  # Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
  probs_ = []
  for idx, prob in enumerate(probs):
      prob = prob.unsqueeze(0)
      probs_.append(torch.cat([prob, torch.zeros([1,prob.shape[1],234 - prob.shape[2]])], dim = 2))
  for idx, prob in enumerate(probs_):
      if probs_[idx].shape[1] < my_config["max_prob_len"]:
          src_padding_mask[idx, prob.shape[1]:] = torch.ones(1, my_config["max_prob_len"] - prob.shape[1])
          probs_[idx] = torch.cat([probs_[idx],torch.Tensor([[[0 if i != 233 else 1 for i in range(234)] for _ in range(my_config["max_prob_len"] - probs_[idx].shape[1])]])],dim=1)

  probs = torch.cat(probs_, dim=0)

  embeddings = list(embeddings)
  for idx, embedding in enumerate(embeddings):
    tgt_padding_mask[idx, embeddings[idx].shape[0] + 1:] = torch.ones(1, my_config["max_embedding_len"] - embeddings[idx].shape[0] - 1)
    embeddings[idx] = embedding.unsqueeze(0)
    embeddings[idx] = torch.cat([embeddings[idx],torch.Tensor([[[0 for i in range(768)] for _ in range(my_config["max_embedding_len"] - embeddings[idx].shape[1])]])],dim=1)
  embeddings = torch.cat(embeddings, dim=0)

  return probs, embeddings, lengths, src_padding_mask, tgt_padding_mask
