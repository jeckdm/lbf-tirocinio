import BF
import SLBF
import LBF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from trainRNN import h_sizes
fprs = [0.001,0.005,0.01,0.02]
fpr_ratios = [0.1*i for i in range(1,11)]
model_sizes = [4,8,16]
loc = "trained_NN/simulations/"      #set location
plot_loc = "trained_NN/plots/"
fpr_ratios2 = [1.*i for i in range(1,11)]



def SLBF_tau_analysis(models,phishing_URLs,verbose,X_train,y_train,device):# Per ognuno dei modelli salvo numero di falsi negativi del classificatore e tau ottimale nelle relative strutture sulla
    # base del fprs e fpr_ratio target.
    false_negs = {}
    taus = {}
    for i in range(3):
        false_negs[i]={}
        taus[i]= {}
        for fpr in fprs:
            for fpr_ratio in fpr_ratios2:
                false_negs[i][(fpr,fpr_ratio)], taus[i][(fpr,fpr_ratio)] = LBF.build_LBF_classifier(models[i], fpr*fpr_ratio,X_train,y_train,device,phishing_URLs)
                if (verbose):
                    print("Modello %d: fpr: %.3f, fpr ratio: %.2f, FNR: %.20f, tau: %.10f" % (i, fpr, fpr_ratio, len(false_negs[i][(fpr,fpr_ratio)])/len(phishing_URLs), taus[i][(fpr,fpr_ratio)]))
                # print(len(false_negs[i][(fpr,fpr_ratio)])/len(phishing_URLs), taus[i][(fpr,fpr_ratio)])
    np.save(loc+"false_negs2", false_negs)
    np.save(loc+"taus2", taus)
    return false_negs,taus


def SLBF_Bloom_filters(models,phishing_URLs,verbose,X_train,y_train,X_test,y_test,testing_list,device):
    SLBF_initials = {}
    SLBF_backups = {}
    # Per ognuno dei modelli salvo il filtro di backup e quello iniziale costruito sulla base del fpr e fpr_ratio target
    try:
        false_negs = np.load(loc+"false_negs2")
        taus = np.load(loc+"taus2")
    except:
        false_negs,taus = SLBF_tau_analysis(models,phishing_URLs,False,X_train,y_train,device)
    for i in range(3):
        SLBF_initials[i] = {}
        SLBF_backups[i] = {}
        for fpr, fpr_ratio in false_negs[i].keys():
            c=(1.-len(false_negs[i][(fpr,fpr_ratio)])/len(phishing_URLs))
            # Se la tau non rispetta i bound
            if(fpr_ratio < c or fpr*fpr_ratio > c):
                print(fpr_ratio, fpr, "bad fpr_tau")
                continue
            # try:
            SLBF_initials[i][(fpr,fpr_ratio)] = SLBF.build_SLBF_initial(false_negs[i][(fpr,fpr_ratio)], fpr, fpr*fpr_ratio)
            SLBF_backups[i][(fpr,fpr_ratio)] = SLBF.build_SLBF_backup(false_negs[i][(fpr,fpr_ratio)], fpr*fpr_ratio)
            if(SLBF_backups[i][(fpr,fpr_ratio)] =='error' or SLBF_initials[i][(fpr,fpr_ratio)]=='error'):
                continue
            fpr0, BF_size, t = SLBF.test_SLBF(SLBF_initials[i][(fpr,fpr_ratio)], models[i], SLBF_backups[i][(fpr,fpr_ratio),], taus[i][(fpr,fpr_ratio)],X_test,y_test)
            if(verbose):
                print(f"teoric fpr: {fpr}, empirc fpr: {fpr0}, size of backup BF: {BF_size}, time : {t}")
            SLBF = {"FPR": fpr0, "size": BF_size+model_sizes[i], "time": t}
            np.save(loc+"SLBF_hid"+str(h_sizes[i])+"_FPR"+str(fpr)+"_ratio"+str(fpr_ratio), SLBF)
