import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from RNN import val,train

def make_batch_train(X_t, y_t, B):
  sample = torch.randint(0, X_t.shape[0], [B]).long()
  batch_X0 = torch.stack([X_t[s] for s in sample])
  batch_X = batch_X0.to(device)
  batch_y0 = torch.stack([y_t[s]*torch.ones(len(X_t[s])).long() for s in sample]).to(device)     #
  batch_y = batch_y0.to(device)
  return batch_X, batch_y
  
def make_batch_test(X_t, y_t, B):
  sample = torch.randint(0, X_t.shape[0], [B]).long()
  batch_X0 = torch.stack([X_t[s] for s in sample])
  batch_X = batch_X0.to(device)
  batch_y0 = torch.stack([y_t[s] for s in sample]).to(device)
  batch_y = batch_y0.to(device)
  return batch_X, batch_y

def give_params():
  emb_size=5
  h_sizes = [16,8,4]
  layers = 1
  criterion = nn.CrossEntropyLoss()
  return emb_size,h_sizes