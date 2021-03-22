import BF
import SLBF
import LBF
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
    Per ognuno dei modelli salvo numero di falsi negativi del classificatore e tau ottimale nelle relative strutture sulla base del fprs e fpr_ratio target.
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
    #np.save(loc+"false_negs2", false_negs) provato per salvare i risultati ( e ri-usarli con SLBF bloom filter ma da' problemi)
    #np.save(loc+"taus2", taus)
    return false_negs,taus


def SLBF_Bloom_filters(models,phishing_URLs,X_train,y_train,X_test,y_test,testing_list,device,verbose=True):
    '''
    Per ognuno dei modelli salvo il filtro di backup e quello iniziale costruito sulla base del fpr e fpr_ratio target
    '''
    
    SLBF_initials = {}
    SLBF_backups = {}
    
   # try:
    #    false_negs = np.load(loc+"false_negs2.npy",allow_pickle=True)
    #    taus = np.load(loc+"taus2.npy",allow_pickle=True)
    #except:  l'idea era di guardare se era già presente un false_negs2 ( prodotto  da tau analysis) ese si caricare quello senza richiamare tau analysis
    #purtroppo l'elemento viene caricato male (problemi con le dimensioni).
    false_negs,taus = SLBF_tau_analysis(models,phishing_URLs,X_train,y_train,device,False)
    for i in range(3):
        SLBF_initials[i] = {}
        SLBF_backups[i] = {}
        for fpr,fpr_ratio in false_negs[i].keys():
            c=(1.-len(false_negs[i][(fpr,fpr_ratio)])/len(phishing_URLs))
            # Se la tau non rispetta i bound
            if(fpr_ratio < c or fpr*fpr_ratio > c):
                print(fpr_ratio, fpr, "bad fpr_tau")
                continue
            # try:
            SLBF_initials[i][(fpr,fpr_ratio)] = SLBF.build_SLBF_initial(false_negs[i][(fpr,fpr_ratio)], fpr, fpr*fpr_ratio,phishing_URLs)
            SLBF_backups[i][(fpr,fpr_ratio)] = SLBF.build_SLBF_backup(false_negs[i][(fpr,fpr_ratio)], fpr*fpr_ratio,phishing_URLs)
            if(SLBF_backups[i][(fpr,fpr_ratio)] =='error' or SLBF_initials[i][(fpr,fpr_ratio)]=='error'):
                continue
            fpr0, BF_size, t = SLBF.test_SLBF(SLBF_initials[i][(fpr,fpr_ratio)], models[i], SLBF_backups[i][(fpr,fpr_ratio)],taus[i][(fpr,fpr_ratio)],X_test,y_test,testing_list)
            if(verbose):
                print(f"teoric fpr: {fpr}, empirc fpr: {fpr0}, size of backup BF: {BF_size}, time : {t}")
            SLBF0 = {"FPR": fpr0, "size": BF_size+model_sizes[i], "time": t}
            np.save(loc+"SLBF_hid"+str(h_sizes[i])+"_FPR"+str(fpr)+"_ratio"+str(fpr_ratio), SLBF0)

def SLBF_graph(models,phishing_URLs,X_train,y_train,X_test,y_test,testing_list,device,falseN=True,FPR=True,Size=True,verbose=True):    #require SLBF_Bloom_filters
    fnrs2 = {}
    # Per ognuno dei modelli costruisco un dataframe in cui salvo il rate di falsi negativi per ogni fpr e fprs_ratio
    #try:
    #    false_negs = np.load(loc+"false_negs2")
    #    taus = np.load(loc+"taus2")
    #except: sesso tentativo fatto in in SLBF_Bloom_filter
    false_negs,taus = SLBF_tau_analysis(models,phishing_URLs,X_train,y_train,device,False)
    SLBF_Bloom_filters(models,phishing_URLs,X_train,y_train,X_test,y_test,testing_list,device,verbose)
    for i in range(3):
        fnrs2[i] = pd.DataFrame(index=fpr_ratios+fpr_ratios2, columns=fprs)
        for fpr,fpr_ratio in false_negs[i].keys():
            fnrs2[i].loc[fpr_ratio,fpr] = len(false_negs[i][(fpr,fpr_ratio)])/len(phishing_URLs)
    true_fpr_SLBF = {}
    sizes_SLBF = {}
    times_SLBF = {}
    # Per ogni modello salvo in base a fpr, fpr_ratio target l'frp empirico, la grandezza ed il tempo di accesso per elemento
    # del SLBF relativo
    # Utile per i grafici successivi
    for i in range(3):
        true_fpr_SLBF[i] = pd.DataFrame(index = fpr_ratios+fpr_ratios2, columns = fprs)
        sizes_SLBF[i] = pd.DataFrame(index = fpr_ratios+fpr_ratios2, columns = fprs)
        times_SLBF[i] = pd.DataFrame(index = fpr_ratios+fpr_ratios2, columns = fprs)
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
        graph(fnrs2,"Classifier False Negative Rate","SLBF_classifier_FNR.png")
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