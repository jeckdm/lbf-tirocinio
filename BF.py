import numpy as np
import torch
from filters.BloomFilter import BloomFilter
import sys
import math
from helpers import *

def run_BF(FPR):
  BF = BloomFilter(len(phishing_URLs), FPR)
  for url in phishing_URLs:
    BF.add(url)
  
  
  fps = 0
  total = 0
  total_time = 0
  for url in testing_list:
    total += 1
    start = time.time()
    result = BF.check(url)
    end = time.time()
    total_time += (end-start)
    if result == True:
      fps += 1
  
  avg_fp = fps/total

  # returns empirical FPR, BF size in bytes, and access time per element
  return avg_fp, BF.size/8, (total_time)/len(testing_list)