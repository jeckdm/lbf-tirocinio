import numpy as np
import torch
import sys
import math

import mmh3
import math
import torch
import time
from bitarray import bitarray 

class BloomFilter(object): 
  
    def __init__(self, items_count,fp_prob): 
        ''' 
        items_count : int 
            Number of items expected to be stored in bloom filter 
        fp_prob : float 
            False Positive probability in decimal 
        '''
        # False posible probability in decimal 
        self.fp_prob = fp_prob 
  
        # Size of bit array to use 
        self.size = self.get_size(items_count,fp_prob) 
  
        # number of hash functions to use 
        self.hash_count = self.get_hash_count(self.size,items_count) 
  
        # Bit array of given size 
        self.bit_array = bitarray(self.size) 
  
        # initialize all bits as 0 
        self.bit_array.setall(0) 
  
    def add(self, item): 
        ''' 
        Add an item in the filter 
        '''
        digests = [] 
        for i in range(self.hash_count): 
            # i works as seed to mmh3.hash() function 
            digest = mmh3.hash(item,i) % self.size 
            digests.append(digest) 
  
            # set the bit True in bit_array 
            self.bit_array[digest] = True
  
    def check(self, item): 
        ''' 
        Check for existence of an item in filter 
        '''
        for i in range(self.hash_count): 
            digest = mmh3.hash(item,i) % self.size 
            if self.bit_array[digest] == False: 
              # if any of bit is False then,its not present 
              # in filter 
              # else there is probability that it exist 
              return False
        return True
  
    @classmethod
    def get_size(self,n,p): 
        ''' 
        Return the size of bit array(m) to used using 
        following formula 
        m = -(n * lg(p)) / (lg(2)^2) 
        n : int 
            number of items expected to be stored in filter 
        p : float 
            False Positive probability in decimal 
        '''
        m = -(n * math.log(p))/(math.log(2)**2) 
        return int(m) 
  
    @classmethod
    def get_hash_count(self, m, n): 
        ''' 
        Return the hash function(k) to be used using 
        following formula 
        k = (m/n) *) 
  
        m : int 
            size of(k)ar 
        n : int 
            number (k)ems exped to be stored in filter 
        '''
        k = (m/n) * math.log(2)
        return int(k)


def run_BF(FPR,phishing_URLs,testing_list):
  BF = BloomFilter(len(phishing_URLs), FPR)
  for url in phishing_URLs:
    BF.add(url)
  
  fps = 0
  total = 0
  total_time = 0
  for urlt in testing_list:
    total += 1
    start = time.time()
    result = BF.check(urlt)
    end = time.time()
    total_time += (end-start)
    if result == True:
      fps += 1
  avg_fp = fps/total
  pino = "ciaopino"
  print(f"avg fp : {fps/total} , fps :{fps}, total: {total}, {BF.check(testing_list[2])}")

  # returns empirical FPR, BF size in bytes, and access time per element
  return avg_fp, BF.size/8, (total_time)/len(testing_list)

def test_size(phishing_URLs,testing_list,loc):    #analizza la size di alcuni bloom filter dato un FPR in una lista
  FPR, BF_size, t = run_BF(0.02,phishing_URLs,testing_list)
  print("FPR", FPR, "size", BF_size, "time", t)
  BF = {"FPR": FPR, "size": BF_size, "time": t}
  np.save(loc+"BF", BF)
  BF_sizes = {}
# Aggiungo alcuni fpr
  fprs = [0.001,0.005,0.01,0.02]
# Stampa grandezza del filtro in relazione al target fpr
  for fpr in fprs:
    BF = BloomFilter(len(phishing_URLs), fpr)
    BF_sizes[fpr] = BF.size / 8
  
  print(BF_sizes)