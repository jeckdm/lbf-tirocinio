import time
from torch.autograd import Variable
from filters.CNN import CNN, trainNet
import numpy as np
import torch
from process import *
from filters.BloomFilter import BloomFilter
import sys
import math
from helpers import *

def build_SLBF_initial(false_negs, FPR, FPR_tau):
  num_false_negs = len(false_negs)
  FPR_B0 = FPR/FPR_tau*(1.-num_false_negs/len(phishing_URLs))
  if(FPR_B0 <= 0 or FPR_B0 >= 1):
    return "error"
  SLBF_initial = BloomFilter(len(phishing_URLs), FPR_B0)
  for url in phishing_URLs:
    SLBF_initial.add(url)
  return SLBF_initial

def build_SLBF_backup(false_negs, FPR_tau):
  num_false_negs = len(false_negs)
  FPR_B = FPR_tau/((1-FPR_tau)*(len(phishing_URLs)/num_false_negs - 1))
  if(FPR_B <= 0):
    return "error"
  SLBF_backup = BloomFilter(num_false_negs, FPR_B)
  for url in false_negs:
    SLBF_backup.add(url)
  return SLBF_backup


def test_SLBF(SLBF_initial, model, SLBF_backup, tau):
  # test on testing data
  fps = 0
  total = 0
  total_time = 0
  for i in range(int(len(y_test)/100)+1):
    x0 = torch.stack([X_test[s] for s in range(100*i, min(100*(i+1), len(y_test)))])
    x = x0.to(device)
    y0 = torch.stack([y_test[s] for s in range(100*i, min(100*(i+1), len(y_test)))])
    y = y0.to(device)  
    total += len(y)
    
    start = time.time()
    y_hat, _ = model(x)
    ps = torch.sigmoid(y_hat[:,:,1])[:,149].squeeze().detach().cpu().numpy()
    for ix, p in enumerate(ps):
      result = SLBF_initial.check(testing_list[100*i+ix])
      if(result):
        if(p>tau):
          result = True
        else:
          result = SLBF_backup.check(testing_list[100*i+ix])
      if(result):
        fps += 1
    end = time.time()
    total_time += (end-start)

  avg_fp = fps/total
  
  # returns empirical FPR, BF size, and avg access time
  return avg_fp, (SLBF_initial.size+SLBF_backup.size) / 8, (total_time)/len(y_test)