import BF
import SLBF
import LBF
import pandas as pd
import numpy as np
from trainRNN import h_sizes
fprs = [0.001,0.005,0.01,0.02]
fpr_ratios = [0.1*i for i in range(1,11)]
model_sizes = [4,8,16]

def BF_test_FPR(teFPR,phishing_URLs,testing_list,loc):   #dato un FPR teorico calcola e stampa l'FPR empirico calcolato usando filtri Bloom.
                                                      #risultato è un dizionario, se chiamato con save = true il risultato 
                                                      #  viene salvato come BF.npy in trained_NN/simulations.
  empFPR, BF_size, t = BF.run_BF(teFPR,phishing_URLs,testing_list)
  print("FPR", FPR, "size", BF_size, "time", t)
  BF = {"FPR": FPR, "size": BF_size, "time": t}
  if save==True:
    np.save(loc+"BF", BF)

def BF_test_size(phishing_URLs,testing_list):  #applica il filtro di bloom ad alcuni fpr pre-stabiliti e stampa size ottenuta
  BF_sizes = {}
# Aggiungo alcuni fpr
# Stampa grandezza del filtro in relazione al target fpr
  for fpr in fprs:
    BFo = BF.BloomFilter(len(phishing_URLs), fpr)
    BF_sizes[fpr] = BFo.size / 8
  print(BF_sizes)


def LBF_tau_analysis(models,phishing_URLs,verbose,X_train,y_train,device):#calcola la tau basandosi su predefiniti fpr e fprs-rate e se verbose=true stampa a video i risultati dell'analisi.
                                               # i falsi negativi e le tau vengono poi ritornate
    false_negs = {}
    taus = {}
    # Per ognuno dei modelli salvo numero di falsi negativi del classificatore e tau ottimale nelle relative strutture sulla
    # base del fprs e fpr_ratio target.
    for i in range(3):
        false_negs[i] = {}
        taus[i] = {}
        for fpr in fprs:
            for fpr_ratio in fpr_ratios:
                false_negs[i][(fpr,fpr_ratio)], taus[i][(fpr,fpr_ratio)] = LBF.build_LBF_classifier(models[i], fpr*fpr_ratio,X_train,y_train,device,phishing_URLs)
                if(verbose):
                    print("Modello %d: %.3f, %.2f, %.20f, %.10f" % (i, fpr, fpr_ratio, len(false_negs[i][(fpr,fpr_ratio)])/len(phishing_URLs), taus[i][(fpr,fpr_ratio)]))
    return false_negs,taus

def save_Backup(models,phishing_URLs,loc,verbose,X_train,y_train,X_test,y_test,testing_list,device):   #salva filtri bloom di backup in trained_NN/anlisys; se verbose = true stampa fpr emirico, size e tempo
                                                                            #dei filtri di backup creati
    LBF_backups = {}
    false_negs,taus = LBF_tau_analysis(models,phishing_URLs,False,X_train,y_train,device)
    # Per ognuno dei modelli salvo il filtro di backup costruito sulla base del fpr e fpr_ratio target
    for i in range(3):
        LBF_backups[i] = {}
    for fpr in fprs:
        for fpr_ratio in fpr_ratios:
            LBF_backups[i][(fpr,fpr_ratio)] = LBF.build_LBF_backup(false_negs[i][(fpr,fpr_ratio)], fpr, fpr*fpr_ratio)
            if(LBF_backups[i][(fpr,fpr_ratio)] =='error'):
                continue
            fpr0, BF_size, t = LBF.test_LBF(models[i], LBF_backups[i][(fpr,fpr_ratio)], taus[i][(fpr,fpr_ratio)],X_test,y_test,testing_list,device)
            print(f"teoric fpr: {fpr}, empirc fpr: {fpr0}, size of backup BF: {BF_size}, time : {t}")
            LBFo = {"FPR": fpr0, "size": BF_size+model_sizes[i], "time": t}
            np.save(loc+"LBF_hid"+str(h_sizes[i])+"_FPR"+str(fpr)+"_ratio"+str(fpr_ratio), LBFo)
            #except:
                # Se il numero di falsi negativi é 0 sollevo eccezione e non salvo
                # Non controllata inizialmente probabilmente perché é stata esclusa la possibilitá di avere fn = 0 con dataset grandi
               # print("Numero falsi negativi = 0")

#def plots_LBF(models,phishing_URLs,loc,verbose,X_train,y_train,device,LBF_back,FalseNeg,FPR,TotSize) # stampa grafici su LBF, i parametri false neg, FPR e totsize determinano  quali grafici stampare