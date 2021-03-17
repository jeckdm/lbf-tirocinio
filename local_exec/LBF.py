from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim
import time
from torch.autograd import Variable
import numpy as np
import torch
from BF import BloomFilter
import sys
import math
from helpers import *
from trainRNN import get_classifier_probs

def determine_tau(FPR_tau, prob_list):
  return np.percentile(np.array(prob_list),100*(1.-FPR_tau))

def build_LBF_classifier(model,FPR_tau,X_train,y_train,device,phishing_URLs):
  probs1, probs0 = get_classifier_probs(model,X_train,y_train,device)
  tau = determine_tau(FPR_tau,probs0)
  false_negs = []
  for i,url in enumerate(phishing_URLs):
    if(probs1[i] < tau):
      false_negs += [url]
  return false_negs, tau


def build_LBF_backup(false_negs, FPR, FPR_tau):
  num_false_negs = len(false_negs)
  FPR_B = (FPR-FPR_tau)/(1-FPR_tau)
  if(FPR_B <= 0):
    return "error"
  LBF_backup = BloomFilter(num_false_negs, FPR_B)
  for url in false_negs:
    LBF_backup.add(url)
  return LBF_backup

def test_LBF(model, LBF_backup, tau,X_test,y_test,testing_list,device):
  # test on testing data
  fps = 0
  total = 0
  total_time = 0
  for i in range(int(len(y_test)/100)+1):
    if( len([X_test[s] for s in range(100*i, min(100*(i+1), len(y_test)))])>0):
      x0 = torch.stack([X_test[s] for s in range(100*i, min(100*(i+1), len(y_test)))])
      x = x0.to(device)
      y0 = torch.stack([y_test[s] for s in range(100*i, min(100*(i+1), len(y_test)))])
      y = y0.to(device)  
      total += len(y)
      
      start = time.time()
      y_hat, _ = model(x)
      ps = torch.sigmoid(y_hat[:,:,1])[:,149].squeeze().detach().cpu().numpy()
      for ix, p in enumerate(ps):
        if(p > tau):
          result = True        
        else:
          result = LBF_backup.check(testing_list[100*i+ix])
        if(result):
          fps += 1
      end = time.time()
      total_time += (end-start)

  avg_fp = fps/total
  
  # returns empirical FPR, BF size, and avg access time
  return avg_fp, LBF_backup.size / 8, (total_time)/len(y_test)
