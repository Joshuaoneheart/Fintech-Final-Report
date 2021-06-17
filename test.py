"""# Inference

## Dataset of inference
"""

from config import my_config
import os
import json
import torch
from pathlib import Path
from torch.utils.data import Dataset

"""## Main funcrion of Inference"""

import json
import csv
from pathlib import Path
from tqdm.notebook import tqdm

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

  dataset = myDataset(my_config["test_phone_dir"], my_config["test_embedding_dir"], "train")
  dataloader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False,
    num_workers=8,
    collate_fn=collate_batch,
  )
  print(f"[Info]: Finish loading data!",flush = True)

  model.load_state_dict(torch.load(my_config["ckpt_name"]))
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
