import BF
import SLBF
import LBF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os.path
from os import path

# Parametri globali
import config

# Rinomino parametri config per comoditá
fprs = config.fprs 
fpr_ratios = config.fpr_ratios2
h_sizes = config.h_sizes 
loc = config.loc_nn 
plot_loc = config.loc_plots
device = config.device


def SLBF_tau_analysis(models,phishing_URLs,X_train,y_train,device,verbose=True): 
    '''
    Per ognuno dei modelli salvo la lista di URL classificati come falsi negativi dal classificatore e tau ottimale nelle relative strutture sulla base del fprs e fpr_ratio target.
    '''

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

    np.save(loc+"false_negs2", false_negs) # provato per salvare i risultati ( e ri-usarli con SLBF bloom filter ma da' problemi)
    np.save(loc+"taus2", taus)

    return false_negs,taus


def SLBF_Bloom_filters(models, phishing_URLs, X_train, y_train, X_test, y_test, testing_list, taus = False, verbose=True):
    '''
    Per ognuno dei modelli salvo il filtro di backup e quello iniziale costruito sulla base del fpr e fpr_ratio target
    '''

    SLBF_initials = {}
    SLBF_backups = {}

    # Evito di rifare analisi di tau se non serve
    if (path.exists(loc + 'false_negs.npy') and taus == False):
        false_negs = np.load(loc + "false_negs.npy", allow_pickle=True)  
        false_negs = false_negs.item() # Ritorna l'item all'interno dell'array caricato, quindi il dizionario

        taus = np.load(loc + "taus.npy", allow_pickle=True)  
        taus = taus.item() # Ritorna l'item all'interno dell'array caricato, quindi il dizionario
    else:
        false_negs, taus = LBF.LBF_tau_analysis(models,phishing_URLs,X_train,y_train, verbose=True)

    for i in range(len(models)):
        SLBF_initials[i] = {}
        SLBF_backups[i] = {}

        for fpr,fpr_ratio in false_negs[i].keys():
            c=(1.-len(false_negs[i][(fpr,fpr_ratio)])/len(phishing_URLs))
            # Se la tau non rispetta i bound
            if(fpr_ratio < c or fpr*fpr_ratio > c):
                print(fpr_ratio, fpr, "bad fpr_tau")
                continue

            try:
                SLBF_initials[i][(fpr,fpr_ratio)] = SLBF.build_SLBF_initial(false_negs[i][(fpr,fpr_ratio)], fpr, fpr*fpr_ratio,phishing_URLs)
                SLBF_backups[i][(fpr,fpr_ratio)] = SLBF.build_SLBF_backup(false_negs[i][(fpr,fpr_ratio)], fpr*fpr_ratio,phishing_URLs)
                if(SLBF_backups[i][(fpr,fpr_ratio)] =='error' or SLBF_initials[i][(fpr,fpr_ratio)]=='error'):
                    continue
                fpr0, BF_size, t = SLBF.test_SLBF(SLBF_initials[i][(fpr,fpr_ratio)], models[i], SLBF_backups[i][(fpr,fpr_ratio)],taus[i][(fpr,fpr_ratio)],X_test,y_test,testing_list)
                if(verbose):
                    print(f"teoric fpr: {fpr}, empirc fpr: {fpr0}, size of backup BF: {BF_size}, time : {t}")

                model_size =  os.path.getsize(loc+"RNN_emb"+str(config.emb_size)+"_hid"+str(h_sizes[i])) # Calcolo size classificatore
                print("SIZE MODELLO: ",  model_size)
                SLBF0 = {"FPR": fpr0, "size": BF_size+model_size, "time": t}
                np.save(loc+"SLBF_hid"+str(h_sizes[i])+"_FPR"+str(fpr)+"_ratio"+str(fpr_ratio), SLBF0)
            except ZeroDivisionError:
                print("Numero falsi negativi = 0")

def SLBF_graph(models, phishing_URLs, X_train, y_train, falseN=True, FPR=True, Size=True, taus = False, verbose=True):    #require SLBF_Bloom_filters
    '''
    Genera grafici su SLBF per ognuno dei (fpr, fprs_ratio)
    Nello specifico genera i seguenti grafici:
    - Andamento FN rate al variare del FPR ratio per ognuno degli FPR target
    - Overall FPR empirico al variare del FPR ratio per ognuno degli FPR target
    - Dimensione totale del SLBF al variare del FPR ratio per ognuno degli FPR target
    I grafici vengono salvati in plot_loc

    Il parametro taus forza l'esecuzione dell'analisi di tau: se é a true i false_negs e taus vengono calcolati invocando LBF_tau_analysis altrimenti vengono caricati 
    dai file taus e false_negs presenti in loc.
    '''

    # Evito di rifare analisi di tau se non serve
    if (path.exists(loc + 'false_negs.npy') and taus == False):
        false_negs = np.load(loc + "false_negs.npy", allow_pickle=True)  
        false_negs = false_negs.item() # Ritorna l'item all'interno dell'array caricato, quindi il dizionario
    else:
        false_negs, _ = LBF.LBF_tau_analysis(models,phishing_URLs,X_train,y_train, verbose=True)

    # Per ognuno dei modelli costruisco un dataframe in cui salvo il rate di falsi negativi per ogni fpr e fprs_ratio
    fnrs = {}

    for i in range(len(models)):
        fnrs[i] = pd.DataFrame(index=fpr_ratios, columns=fprs)
        for fpr,fpr_ratio in false_negs[i].keys():
            fnrs[i].loc[fpr_ratio,fpr] = len(false_negs[i][(fpr,fpr_ratio)])/len(phishing_URLs)

    # Per ogni modello salvo in base a fpr, fpr_ratio target l'frp empirico, la grandezza ed il tempo di accesso per elemento del SLBF relativo
    # Utile per i grafici successivi
    true_fpr_SLBF = {}
    sizes_SLBF = {}
    times_SLBF = {}

    for i in range(len(models)):
        true_fpr_SLBF[i] = pd.DataFrame(index = fpr_ratios, columns = fprs)
        sizes_SLBF[i] = pd.DataFrame(index = fpr_ratios, columns = fprs)
        times_SLBF[i] = pd.DataFrame(index = fpr_ratios, columns = fprs)
        for fpr,fpr_ratio in false_negs[i].keys():
            try:
                SLBF = np.load(loc+"SLBF_hid"+str(h_sizes[i])+"_FPR"+str(fpr)+"_ratio"+str(fpr_ratio)+".npy", allow_pickle=True).item()
                true_fpr_SLBF[i].loc[fpr_ratio,fpr] = SLBF['FPR']
                sizes_SLBF[i].loc[fpr_ratio,fpr] = SLBF['size']
                times_SLBF[i].loc[fpr_ratio,fpr] = SLBF['time']
                print(f"""{SLBF['FPR']}, {SLBF['size']}, {SLBF['time']}""")
            except:
                print("error", fpr_ratio, fpr) # Bad tau + false negs = 0
                continue

    if(falseN):
        graph(fnrs,"Classifier False Negative Rate","SLBF_classifier_FNR.png")
    if(FPR):
        graph(true_fpr_SLBF,"Overall FPR","SLBF_fpr.png")
    if(FPR):
        graph(sizes_SLBF,"Total Size of SLBF","SLBF_size.png")
        
    

def graph(params,title,path):
    f,ax = plt.subplots(1,3,figsize=(12,3))
    for i in range(3):
        params[i].plot(ax=ax[i])
        ax[i].legend(fontsize='xx-small')
        ax[i].set_xlabel("Classifier FPR ratio")
        ax[i].set_ylabel(title)
        ax[i].set_title("SLBF with "+str(h_sizes[i])+" dimensional GRU")
    plt.tight_layout()
    #plt.show()
    f.savefig(plot_loc+path)
    # FPR_tau/FPR forced to stay between 0 and 1, 

#def FPR_ratio(true_fpr_SLBF):   generalizzata la funzione graph. chiedo se una buona idea, altrimenti ripristino queste
#   f,ax=plt.subplots(1,3,figsize=(12,3))
#    for i in range(3):
#        true_fpr_SLBF[i].plot(ax=ax[i])
#        ax[i].legend(fontsize='xx-small')
#        ax[i].set_xlabel("Classifier FPR ratio")
#        ax[i].set_ylabel("Overall FPR")
#        ax[i].set_title("SLBF with "+str(h_sizes[i])+" dimensional GRU")
#        plt.tight_layout()
    #plt.show()
#    f.savefig(plot_loc+"SLBF_fpr.png")


#def size_ratio(sizes_SLBF):
#    f, ax = plt.subplots(1,3,figsize=(12,3))
#    for i in range(3):
#        sizes_SLBF[i].plot(ax=ax[i])
#        ax[i].legend(fontsize='xx-small')
#        ax[i].set_xlabel("Classifier FPR ratio")
#        ax[i].set_ylabel("Total Size of SLBF")
#        ax[i].set_title("SLBF with "+str(h_sizes[i])+" dimensional GRU")
#    plt.tight_layout()
#    plt.show()
#    f.savefig(plot_loc+"SLBF_size.png")


#size is best when ratio = 1

def SLBF_total_analisys(models,phishing_URLs,X_train,y_train,X_test,y_test,testing_list,device,verbose):
    SLBF_Bloom_filters(models,phishing_URLs,X_train,y_train,X_test,y_test,testing_list,device,verbose)
    SLBF_graph(models,phishing_URLs,X_train,y_train,X_test,y_test,testing_list,device,verbose=verbose)