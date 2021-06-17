"""# Inference

## Dataset of inference
"""

from config import my_config
import os
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset
from model.transformer import Seq_Encode
from config import my_config

"""## Main funcrion of Inference"""

import json
import csv
from pathlib import Path
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from utils.data import myDataset, collate_batch

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

  dataset = myDataset(my_config, "test")
  dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=8,
    collate_fn=collate_batch,
  )
  print(f"[Info]: Finish loading data!",flush = True)
  model = Seq_Encode(my_config, device = device).to(device)

  model.load_state_dict(torch.load(my_config["ckpt_name"]))
  model.eval()
  print(f"[Info]: Finish creating model!",flush = True)

  results = [["Id", "Category"]]
  for mels, labels, lengths, src_padding_mask, tgt_padding_mask in tqdm(dataloader):
    with torch.no_grad():
      mels = mels.to(device)
      labels = labels.to(device)
      src_padding_mask = src_padding_mask.to(device).bool()
      tgt_padding_mask = tgt_padding_mask.to(device).bool()
      outs = model.generate(mels, labels, lengths, device, src_padding_mask, tgt_padding_mask)


if __name__ == "__main__":
  main(**parse_args())
