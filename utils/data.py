class myDataset(Dataset):
  def __init__(self, phone_dir, embedding_dir, split="test"):
    # data: [batch, label]
    self.phone_dir = phone_dir
    self.embedding_dir = embedding_dir
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
        size += 1
        self.data.append((idx, int(start), int(end)+1))
    print(f"Data size: {size}")      
    
  def __len__(self):
    return len(self.data)

  def __getitem__(self, index):
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

"""## Dataloader
- Split dataset into training dataset(90%) and validation dataset(10%).
- Create dataloader to iterate the data.

"""

import torch
from torch.utils.data import DataLoader, random_split


def collate_batch(batch):
  # Process features within a batch.
  """Collate a batch of data."""
  probs, embeddings, lengths = zip(*batch)
  # Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
  probs_ = []
  src_masks = []
  for idx, prob in enumerate(probs):
      prob = prob.unsqueeze(0)
      probs_.append(torch.cat([prob, torch.zeros([1,prob.shape[1],234 - prob.shape[2]])], dim = 2))
  for idx, prob in enumerate(probs_):
      if probs_[idx].shape[1] < my_config["max_prob_len"]:
          probs_[idx] = torch.cat([probs_[idx],torch.Tensor([[[0 if i != 233 else 1 for i in range(234)] for _ in range(my_config["max_prob_len"] - probs_[idx].shape[1])]])],dim=1)
      else:
          probs_[idx] = probs_[idx][:,:my_config["max_prob_len"],:]
  probs = torch.cat(probs_, dim=0)

  embeddings = list(embeddings)
  for idx, embedding in enumerate(embeddings):
    embeddings[idx] = embedding.unsqueeze(0)
    embeddings[idx] = torch.cat([embeddings[idx],torch.Tensor([[[0 for i in range(768)] for _ in range(my_config["max_embedding_len"] - embeddings[idx].shape[1])]])],dim=1)
  embeddings = torch.cat(embeddings, dim=0)

  return probs, embeddings, lengths
