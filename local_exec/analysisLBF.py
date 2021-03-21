import BF
import LBF
import pandas as pd
import numpy as np
from trainRNN import h_sizes
import matplotlib.pyplot as plt

fprs = [0.001,0.005,0.01,0.02]
fpr_ratios = [0.1*i for i in range(1,11)]
model_sizes = [4,8,16]
loc = "trained_NN/simulations/"      #set location
plot_loc = "trained_NN/plots/"



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
                    print("Modello %d: fpr: %.3f, fpr ratio: %.2f, FNR: %.20f, %.10f" % (i, fpr, fpr_ratio, len(false_negs[i][(fpr,fpr_ratio)])/len(phishing_URLs), taus[i][(fpr,fpr_ratio)]))
    np.save(loc+"false_negs", false_negs)
    np.save(loc+"taus", taus)
    return false_negs,taus

def save_Backup(models,phishing_URLs,verbose,X_train,y_train,X_test,y_test,testing_list,device):   #salva filtri bloom di backup in trained_NN/anlisys; se verbose = true stampa fpr emirico, size e tempo
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
            if(verbose):
              print(f"teoric fpr: {fpr}, empirc fpr: {fpr0}, size of backup BF: {BF_size}, time : {t}")
            LBFo = {"FPR": fpr0, "size": BF_size+model_sizes[i], "time": t}
            np.save(loc+"LBF_hid"+str(h_sizes[i])+"_FPR"+str(fpr)+"_ratio"+str(fpr_ratio), LBFo)
            #except:
                # Se il numero di falsi negativi é 0 sollevo eccezione e non salvo
                # Non controllata inizialmente probabilmente perché é stata esclusa la possibilitá di avere fn = 0 con dataset grandi
               # print("Numero falsi negativi = 0")

  #def plots_LBF(models,phishing_URLs,loc,verbose,X_train,y_train,device,LBF_back,FalseNeg,FPR,TotSize) # stampa grafici su LBF, i parametri false neg, FPR e totsize determinano  quali grafici stampare
def LBF_graph(models,phishing_URLs,X_train,y_train,device,falseN,FPR,Size,verbose):
  false_negs,_=LBF_tau_analysis(models,phishing_URLs,False,X_train,y_train,device)
  fnrs = {}
  # Per ognuno dei modelli costruisco un dataframe in cui salvo il rate di falsi negativi per ogni fpr e fprs_ratio
  for i in range(3):
    fnrs[i] = pd.DataFrame(index=fpr_ratios, columns=fprs)
    for fpr in fprs:
      for fpr_ratio in fpr_ratios:
        fnrs[i].loc[fpr_ratio,fpr] = len(false_negs[i][(fpr,fpr_ratio)])/len(phishing_URLs)
  true_fpr_LBF = {}
  sizes_LBF = {}
  times_LBF = {}
  # Per ogni modello salvo in base a fpr, fpr_ratio target l'frp empirico, la grandezza ed il tempo di accesso per elemento
  # del LBF relativa
  # Utile per i grafici successivi
  for i in range(3):
    true_fpr_LBF[i] = pd.DataFrame(index = fpr_ratios, columns = fprs)
    sizes_LBF[i] = pd.DataFrame(index = fpr_ratios, columns = fprs)
    times_LBF[i] = pd.DataFrame(index = fpr_ratios, columns = fprs)
    for fpr in fprs:
      for fpr_ratio in fpr_ratios:
        try:
          LBF = np.load(loc+"LBF_hid"+str(h_sizes[i])+"_FPR"+str(fpr)+"_ratio"+str(fpr_ratio)+".npy", allow_pickle=True).item()
          true_fpr_LBF[i].loc[fpr_ratio,fpr] = LBF['FPR']
          sizes_LBF[i].loc[fpr_ratio,fpr] = LBF['size']
          times_LBF[i].loc[fpr_ratio,fpr] = LBF['time']
          print(f"""{LBF['FPR']}, {LBF['size']}, {LBF['time']}""")
        except:
          # Aggiunta except utile nel caso in cui il file non fosse stato salvato perché fn = 0
          print("error / numero falsi negativi 0")
          continue
  if(falseN):
    falseN_ratio_graph(fnrs)
  if(FPR):
    FPR_ratio_graph(true_fpr_LBF)
  if(Size):
    size_ratio_graph(sizes_LBF)

def falseN_ratio_graph(fnrs):
  f,ax = plt.subplots(1,3,figsize=(12,3))
  for i in range(3):
    fnrs[i].plot(ax=ax[i])
    ax[i].set_xlabel("Classifier FPR ratio "+r"$\epsilon_\tau/\epsilon$")
    ax[i].set_ylabel("Classifier False Negative Rate")
    ax[i].set_title("LBF with "+str(h_sizes[i])+" dimensional GRU")
    ax[i].legend(fontsize='xx-small')
  plt.tight_layout()
  #plt.show()
  f.savefig(plot_loc+"LBF_classifier_FNR.png")
  # FPR_tau/FPR forced to stay between 0 and 1, 

def FPR_ratio_graph(true_fpr_LBF):
  f, ax = plt.subplots(1,3,figsize=(12,3))
  for i in range(3):
    true_fpr_LBF[i].plot(ax=ax[i])
    ax[i].set_xlabel("Classifier FPR ratio "+r"$\epsilon_\tau/\epsilon$")
    ax[i].set_ylabel("Overall FPR")
    ax[i].set_title("LBF with "+str(h_sizes[i])+" dimensional GRU")
    ax[i].legend(fontsize='xx-small', loc='upper right')
  plt.tight_layout()
  #plt.show()
  f.savefig(plot_loc+"LBF_fpr.png")

def size_ratio_graph(sizes_LBF):
  f, ax = plt.subplots(1,3,figsize=(12,3))
  for i in range(3):
    sizes_LBF[i].plot(ax=ax[i])
    ax[i].set_xlabel("Classifier FPR ratio "+r"$\epsilon_\tau/\epsilon$")
    ax[i].set_ylabel("Total Size of LBF")
    ax[i].set_title("LBF with "+str(h_sizes[i])+" dimensional GRU")
    ax[i].legend(fontsize='xx-small', loc='upper left')
  plt.tight_layout()
  #plt.show()
  f.savefig(plot_loc+"LBF_size.png")
# seems most optimal at 0.5

def total_LBF_analisys(models,phishing_URLs,verbose,X_train,y_train,X_test,y_test,testing_list,device):
  save_Backup(models,phishing_URLs,verbose,X_train,y_train,X_test,y_test,testing_list,device)
  LBF_graph(models,phishing_URLs,X_train,y_train,device,falseN,FPR,Size,True,True,True)