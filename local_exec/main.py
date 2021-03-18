# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import init    #pre-processing -> see init.py
import trainRNN
from trainRNN  import emb_size,h_sizes #parametri per training della RNN -> per info train.py
import analysis
#interfaces
def load_RNN():
    return trainRNN.load(X_train,y_train,loc,device)
def BF_test_size():
    analysis.BF_test_size(phishing_URLs,testing_list)
def LBF_tau_analysis(verbose):#calcola la tau basandosi su predefiniti fpr e fprs-rate e se verbose=true stampa a video i risultati dell'analisi.
    analysis.LBF_tau_analysis(models,phishing_URLs,verbose,X_train,y_train,device)
def save_Backup(models,phishing_URLs,loc,verbose,X_train,y_train,X_test,y_test,testing_list,device):   #salva filtri bloom di backup in trained_NN/anlisys; se verbose = true stampa fpr emirico, size e tempo
                                                                            #dei filtri di backup creati

device,accelerator = init.GPU_init()    # define device and accelerator for GPU acceleration (NVIDIA) 
                                            #(if you don't have NVIDA training will lose in performance)
legitimate_URLs,phishing_URLs = init.take_input() #first pre processing on input
X_train,y_train,X_test,y_test,training_list,testing_list = init.get_train_set(legitimate_URLs,phishing_URLs)  #return training set; see init.py for details

loc = "trained_NN/simulations/"      #set location
plot_loc = "trained_NN/plots/"

#trainRNN.train(X_train,y_train,loc,device) #eseguire solo la prima volta

models = load_RNN()
analysis.save_Backup(models,phishing_URLs,loc,True,X_train,y_train,X_test,y_test,training_list,device)
