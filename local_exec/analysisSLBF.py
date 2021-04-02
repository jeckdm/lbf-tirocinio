import BF
import SLBF
import LBF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
from os import path
import helpers
import analysisTau
import graph
# Parametri globali
import config

# Rinomino parametri config per comoditá
device = config.device

'''
Per ognuno dei modelli salvo il filtro di backup e quello iniziale costruito sulla base del fpr e fpr_ratio target

Il parametro taus forza l'esecuzione dell'analisi di tau: se é a true i false_negs e taus vengono calcolati invocando LBF_tau_analysis altrimenti vengono caricati 
dai file taus e false_negs presenti in loc.
Il parametro name contiene la coppia che raprresenta i nomi dei file da cui vengono caricati il il dizionario (fpr, fpr_ratio): falsi negativi e (fpr, fpr_ratio): tau
se tali file non esistono vengono creati con il nome indicato in name tramite una chiamata a helpers.tau_analysis
'''
def create_SLBF_filters(models, fprs, fpr_ratios, false_negs, training_list): # Forse training list non va bene? Dovrei mettere phishing URLs?
    SLBF_initials = {}
    SLBF_backups = {}

    for i in range(len(models)):
        SLBF_initials[i] = {}
        SLBF_backups[i] = {}

        for fpr,fpr_ratio in false_negs[i].keys():
            c=(1.-len(false_negs[i][(fpr,fpr_ratio)])/len(training_list))
            # Se la tau non rispetta i bound
            if(fpr_ratio < c or fpr*fpr_ratio > c):
                print(fpr_ratio, fpr, "bad fpr_tau")
                continue
            try:
                SLBF_initials[i][(fpr,fpr_ratio)] = SLBF.build_SLBF_initial(false_negs[i][(fpr,fpr_ratio)], fpr, fpr*fpr_ratio,training_list)
                SLBF_backups[i][(fpr,fpr_ratio)] = SLBF.build_SLBF_backup(false_negs[i][(fpr,fpr_ratio)], fpr*fpr_ratio,training_list)
                if(SLBF_backups[i][(fpr,fpr_ratio)] =='error' or SLBF_initials[i][(fpr,fpr_ratio)]=='error'):
                    continue
            except ZeroDivisionError:
                print("Numero falsi negativi = 0")
        
    return SLBF_initials, SLBF_backups

'''
'''
def empirical_analysis(models, fprs, fpr_ratios, SLBFs, X_test, y_test, testing_list, taus, save = False):
    # Per ogni modello salvo in base a fpr, fpr_ratio target l'frp empirico, la grandezza ed il tempo di accesso per elemento del SLBF relativo
    # Utile per i grafici successivi
    true_fpr_SLBF = {}
    sizes_SLBF = {}
    times_SLBF = {}
    size_struct_SLBF = {}
    structs = ["intial_BF","model","backup_BF"]
    Fpr_Const = 0.02   #Fpr per il quale vado a testare le size del classificatore e di BF filter; possibile parametro
    # Grandezze dei modelli addestrati (Dei soli parametri (?))
    models_size = analysisTau.get_models_size()
    SLBFs_initial,SLBFs_backup = SLBFs
    for i in range(len(models)):
        true_fpr_SLBF[i] = pd.DataFrame(index = fpr_ratios, columns = fprs)
        sizes_SLBF[i] = pd.DataFrame(index = fpr_ratios, columns = fprs)
        times_SLBF[i] = pd.DataFrame(index = fpr_ratios, columns = fprs)
        size_struct_SLBF[i] = pd.DataFrame(index = fpr_ratios, columns = structs)
        model_size = models_size[i]
        for fpr in fprs:
            for fpr_ratio in fpr_ratios:
                try:
                    # Carico il relativo SLBF/LBF
                    #SLBF_filters = SLBFs[i][(fpr,fpr_ratio)] tupla,non credo possa funzionare
                    BF_initial = SLBFs_initial[i][(fpr,fpr_ratio)]
                    BF_backup = SLBFs_backup[i][(fpr,fpr_ratio)]
                    # Calcolo parametri empirici
                    fpr0, SLBF_size, t = SLBF.test_SLBF(BF_initial, models[i], BF_backup, taus[i][(fpr,fpr_ratio)],X_test,y_test,testing_list)
                    # Calcolo la size del modello
                    # Salvo i risultati
                    true_fpr_SLBF[i].loc[fpr_ratio,fpr] = fpr0
                    sizes_SLBF[i].loc[fpr_ratio,fpr] = SLBF_size + model_size
                    times_SLBF[i].loc[fpr_ratio,fpr] = t
                    print(f"FPR Target, FPR Ratio: ({fpr},{fpr_ratio}), FPR empirico: {fpr0}, Size totale: {BF_size + model_size}, Size modello: {model_size} Tempo di accesso medio: {t}")
                    if save:
                        SLBFo = {"FPR": fpr0, "size": SLBF_size+model_size, "time": t}
                        # Andrebbe cambiato formato di salvataggio
                        # np.save(config.loc_nn+"SLBF_hid"+str(config.h_sizes[i])+"_FPR"+str(fpr)+"_ratio"+str(fpr_ratio), LBFo)
                except:
                    print("error", fpr_ratio, fpr) # Bad tau + false negs = 0
                    continue
        for fpr_ratio in fpr_ratios:
            try:
                BF_backup = SLBFs_backup[i][(Fpr_Const,fpr_ratio)]
                BF_initial = SLBFs_backup[i][(Fpr_Const,fpr_ratio)]
                size_struct_SLBF[i].loc[fpr_ratio,"backup_BF"] = BF_backup.size/8
                size_struct_SLBF[i].loc[fpr_ratio,"initial_BF"] = BF_initial.size/8
                size_struct_SLBF[i].loc[fpr_ratio,"model"] = model_size[i]
            except:
                continue

    return true_fpr_SLBF, sizes_SLBF, times_SLBF,size_struct_SLBF

def SLBF_total_analisys(models, fprs, fpr_ratios, training_list, X_train, y_train, X_test, y_test, testing_list, verbose=False):
    # Faccio analisi di tau e salvo relativi file
    print("ANALISI TAU")
    false_negs, taus = analysisTau.tau_analysis(models, fprs, fpr_ratios, training_list, X_train, y_train, name=("false_negs2", "taus2"))
    # false_negs = np.load(config.loc_nn + "false_negs.npy", allow_pickle=True).item()
    # taus = np.load(config.loc_nn + "taus.npy", allow_pickle=True).item()
    # Creo i filtri di backup sulla base di fprs, fpr_ratios
    print("CREAZIONI BF BACKUP")
    SLBF_filters = create_SLBF_filters(models, fprs, fpr_ratios, false_negs, training_list)
    # Calcolo rate di falsi negativi per ogni fprs, fpr_ratios
    print("CALCOLO FNRS")
    fnrs = analysisTau.fnrs_analysis(models, fprs, fpr_ratios, false_negs, training_list)
    # Analisi empirica delle strutture create
    print("ANALISI EMPIRICA")
    true_fpr_SLBF, sizes_SLBF, times_SLBF, sizes_struct_SLBF = empirical_analysis(models, fprs, fpr_ratios, SLBF_filters, X_test, y_test, testing_list, taus)
    # Genero grafici
    graph.LBF_graph(fnrs, true_fpr_SLBF, sizes_SLBF,size_struct_SLBF, "SLBF")
    
