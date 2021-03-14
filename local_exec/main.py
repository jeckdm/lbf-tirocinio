import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import init    #pre-processing -> see init.py
import trainRNN  #parametri per training della RNN -> per info train.py
# Libraries
device,accelerator = init.GPU_init()    # define device and accelerator for GPU acceleration (NVIDIA) 
                                            #(if you don't have NVIDA training will lose in performance)
legitimate_URLs,phishing_URLs = init.take_input() #first pre processing on input

X_train,y_train,X_test,y_test = init.get_train_set(legitimate_URLs,phishing_URLs)  #return training set; see init.py for details

loc = "trained_NN/simulations/"      #set location
plot_loc = "trained_NN/plots/"

emb_size,h_sizes = trainRNN.give_params()