# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import seaborn as sns
import init    #pre-processing -> see init.py
import trainRNN
from trainRNN  import emb_size,h_sizes #parametri per training della RNN -> per info train.py
import analysisBF
import analysisLBF
#interfaces
def load_RNN():
    return trainRNN.load(X_train,y_train,loc,device)
def BF_test_size():
    analysis.BF_test_size(phishing_URLs,testing_list)
def LBF_tau_analysis(verbose=False):#calcola la tau basandosi su predefiniti fpr e fprs-rate e se verbose=true stampa a video i risultati dell'analisi.
    analysisLBF.LBF_tau_analysis(models,phishing_URLs,verbose,X_train,y_train,device)
def save_Backup(verbose=False):
    analysisLBF.save_Backup(models,phishing_URLs,verbose,X_train,y_train,X_test,y_test,testing_list,device)
                     #salva filtri bloom di backup in trained_NN/anlisys; se verbose = true stampa fpr emirico, size e tempo
                                                                            #dei filtri di backup creati
def LBF_graph(falseN,FPR,Size,verbose=False):
    analysisLBF.LBF_graph(models,phishing_URLs,X_train,y_train,device,falseN,FPR,Size,verbose)
def SLBF_Bloom_filters_analysis(verbose):
    analysisSLBF.SLBF_Bloom_filters_analysis(models,phishing_URLs,verbose,X_train,y_train,X_test,y_test,testing_list,device)



device,accelerator = init.GPU_init()    # define device and accelerator for GPU acceleration (NVIDIA) 
                                            #(if you don't have NVIDA training will lose in performance)
legitimate_URLs,phishing_URLs = init.take_input() #first pre processing on input
X_train,y_train,X_test,y_test,training_list,testing_list = init.get_train_set(legitimate_URLs,phishing_URLs)  #return training set; see init.py for details

loc = "trained_NN/simulations/"      #set location
plot_loc = "trained_NN/plots/"

#trainRNN.train(X_train,y_train,loc,device) #eseguire solo la prima volta

models = load_RNN()
#LBF_graph(True,True,True)
SLBF_Bloom_filters_analysis(True)