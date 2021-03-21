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
import analysisSLBF
#interfaces
def load_RNN():
    return trainRNN.load(X_train,y_train,loc,device)
#BF
def BF_test_size():
    analysis.BF_test_size(phishing_URLs,testing_list)
#LBF analysis
def LBF_tau_analysis(verbose=True):#calcola la tau basandosi su predefiniti fpr e fprs-rate e se verbose=true stampa a video i risultati dell'analisi.
    analysisLBF.LBF_tau_analysis(models,phishing_URLs,verbose,X_train,y_train,device)
def save_Backup(verbose=True): #salva filtri bloom di backup in trained_NN/analisys; se verbose = true stampa fpr emirico, size e tempo
    analysisLBF.save_Backup(models,phishing_URLs,X_train,y_train,X_test,y_test,testing_list,device,verbose)
def LBF_graph(falseN=True,FPR=True,Size=True,verbose=False): #stampa grafici, specifico quali grafici (default tutti) e se stampare msg (default true). richiede esecuzione precedente di save Backup
    analysisLBF.LBF_graph(models,phishing_URLs,X_train,y_train,device,falseN,FPR,Size,verbose)
def LBF_total_analisys(verbose = True): #fa analisi completa del lbf, va a eseguire internamente tau analisis, elabora e salva i backup filter e crea i grafici
    analysisLBF.total_LBF_analisys(models,phishing_URLs,X_train,y_train,X_test,y_test,testing_list,device,verbose)

#SLBF analysis
def SLBF_tau_analysis(verbose = True):
    analysisSLBF.SLBF_tau_analysis(models,phishing_URLs,X_train,y_train,device,verbose)
def SLBF_Bloom_filters_analysis(verbose=True): #crea Bloom filters necessari per SLBF e li salva in TrainNN/analysis
    analysisSLBF.SLBF_Bloom_filters(models,phishing_URLs,X_train,y_train,X_test,y_test,testing_list,device,verbose)
def SLBF_graph(falseN=True,FPR=True,Size=True,verbose=False): #vedi LBF_graph
    analysisSLBF.SLBF_graph(models,phishing_URLs,X_train,y_train,X_test,y_test,testing_list,device,falseN=True,FPR=True,Size=True,verbose=True)
def SLBF_total_analisys(verbose=True):
    analysisSLBF.SLBF_total_analisys(models,phishing_URLs,X_train,y_train,X_test,y_test,testing_list,device,verbose)

loc = "trained_NN/simulations/"      #set location
plot_loc = "trained_NN/plots/"

device,accelerator = init.GPU_init()    # define device and accelerator for GPU acceleration (NVIDIA) 
                                            #(if you don't have NVIDA training will lose in performance)
legitimate_URLs,phishing_URLs = init.take_input() #first pre processing on input
X_train,y_train,X_test,y_test,training_list,testing_list = init.get_train_set(legitimate_URLs,phishing_URLs)  #return training set; see init.py for details
#trainRNN.train(X_train,y_train,loc,device) #eseguire solo la prima volta
models = load_RNN()
#analysis function
LBF_total_analisys(False)
SLBF_total_analisys(False)