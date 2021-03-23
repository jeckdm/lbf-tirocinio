import BF
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
fpr_ratios = config.fpr_ratios 
h_sizes = config.h_sizes 
loc = config.loc_nn 
plot_loc = config.loc_plots
device = config.device

def LBF_tau_analysis(models,phishing_URLs,X_train,y_train,verbose = True):
  '''
  calcola la tau basandosi su predefiniti fpr e fprs-rate e se verbose=true stampa a video i risultati dell'analisi.
  i falsi negativi e le tau relativi ad ogni (fpr, fprs-rate) sono salvati come false_negs.npy e taus.npy in loc e vengono poi ritornati dalla funzione
  '''
  
  false_negs = {}
  taus = {}
  # Per ognuno dei modelli salvo numero di falsi negativi del classificatore e tau ottimale nelle relative strutture sulla
  # base del fprs e fpr_ratio target.
  for i in range(len(models)): # Cambiato range da 3 a models
    false_negs[i] = {}
    taus[i] = {}
    for fpr in fprs:
      for fpr_ratio in fpr_ratios:
        false_negs[i][(fpr,fpr_ratio)], taus[i][(fpr,fpr_ratio)] = LBF.build_LBF_classifier(models[i], fpr*fpr_ratio,X_train,y_train,phishing_URLs)
        if(verbose):
          print("Modello %d: fpr: %.3f, fpr ratio: %.2f, FNR: %.20f, %.10f" % (i, fpr, fpr_ratio, len(false_negs[i][(fpr,fpr_ratio)])/len(phishing_URLs), taus[i][(fpr,fpr_ratio)]))

  np.save(loc+"false_negs", false_negs)
  np.save(loc+"taus", taus)

  return false_negs,taus

def save_Backup(models, phishing_URLs, X_train, y_train, X_test, y_test, testing_list, taus = False, verbose=True):   
  '''
  salva filtri bloom di backup in trained_NN/anlisys; se verbose = true stampa fpr empirico, size e tempo dei filtri di backup creati

  Il parametro taus forza l'esecuzione dell'analisi di tau: se é a true i false_negs e taus vengono calcolati invocando LBF_tau_analysis altrimenti vengono caricati 
  dai file taus e false_negs presenti in loc.
  '''

  # Evito di rifare analisi di tau se non serve
  if (path.exists(loc + 'false_negs.npy') and taus == False):
    false_negs = np.load(loc + "false_negs.npy", allow_pickle=True)  
    false_negs = false_negs.item() # Ritorna l'item all'interno dell'array caricato, quindi il dizionario

    taus = np.load(loc + "taus.npy", allow_pickle=True)  
    taus = taus.item() # Ritorna l'item all'interno dell'array caricato, quindi il dizionario
  else:
    false_negs, taus = LBF_tau_analysis(models,phishing_URLs,X_train,y_train, verbose=True)

  LBF_backups = {}

  # Per ognuno dei modelli salvo il filtro di backup costruito sulla base del fpr e fpr_ratio target
  for i in range(len(models)): # Cambiato range da 3 a models
    LBF_backups[i] = {}
    for fpr in fprs:
        for fpr_ratio in fpr_ratios:
          try:
            LBF_backups[i][(fpr,fpr_ratio)] = LBF.build_LBF_backup(false_negs[i][(fpr,fpr_ratio)], fpr, fpr*fpr_ratio)
            if(LBF_backups[i][(fpr,fpr_ratio)] =='error'):
                continue
            fpr0, BF_size, t = LBF.test_LBF(models[i], LBF_backups[i][(fpr,fpr_ratio)], taus[i][(fpr,fpr_ratio)],X_test,y_test,testing_list)
            if(verbose):
              print(f"teoric fpr: {fpr}, empirc fpr: {fpr0}, size of backup BF: {BF_size}, time : {t}")
            model_size =  os.path.getsize(loc+"RNN_emb"+str(config.emb_size)+"_hid"+str(h_sizes[i])) # Calcolo size classificatore
            print("SIZE MODELLO: ",  model_size)
            LBFo = {"FPR": fpr0, "size": BF_size+model_size, "time": t}
            np.save(loc+"LBF_hid"+str(h_sizes[i])+"_FPR"+str(fpr)+"_ratio"+str(fpr_ratio), LBFo)
          except ZeroDivisionError:
              # Se il numero di falsi negativi é 0 sollevo eccezione e non salvo
              # Non controllata inizialmente probabilmente perché é stata esclusa la possibilitá di avere fn = 0 con dataset grandi
              print("Numero falsi negativi = 0")

def LBF_graph(models, phishing_URLs, X_train, y_train, taus=False, falseN=True, FPR=True, Size=True, verbose=True):
  '''
  Genera grafici su LBF per ognuno dei (fpr, fprs_ratio)
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
    false_negs, _ = LBF_tau_analysis(models,phishing_URLs,X_train,y_train, verbose=True)

  # Per ognuno dei modelli costruisco un dataframe in cui salvo il rate di falsi negativi per ogni fpr e fprs_ratio
  fnrs = {}
  for i in range(len(models)): # Cambiato range da 3 a models
    fnrs[i] = pd.DataFrame(index=fpr_ratios, columns=fprs)
    for fpr in fprs:
      for fpr_ratio in fpr_ratios:
        fnrs[i].loc[fpr_ratio,fpr] = len(false_negs[i][(fpr,fpr_ratio)])/len(phishing_URLs)


  # Per ogni modello salvo in base a fpr, fpr_ratio target l'frp empirico, la grandezza ed il tempo di accesso per elemento del LBF relativa
  # Utile per i grafici successivi
  true_fpr_LBF = {}
  sizes_LBF = {}
  times_LBF = {}

  for i in range(len(models)): # Cambiato range da 3 a models
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
    graph(fnrs, "Classifier False Negative Rate", "LBF_classifier_FNR.png", )
  if(FPR):
    graph(true_fpr_LBF, "Overall FPR", "LBF_fpr.png", )
  if(Size):
    graph(sizes_LBF, "Total Size of LBF",  "LBF_size.png")


def graph(params,title,path):
  f,ax = plt.subplots(1, len(h_sizes),figsize=(12,3))
  for i in range(len(h_sizes)):
    params[i].plot(ax=ax[i])
    ax[i].set_xlabel("Classifier FPR ratio "+r"$\epsilon_\tau/\epsilon$")
    ax[i].set_ylabel(title)
    ax[i].set_title("LBF with "+str(h_sizes[i])+" dimensional GRU")
    ax[i].legend(fontsize='xx-small')
  plt.tight_layout()
  #plt.show()
  f.savefig(plot_loc+path)
  # FPR_tau/FPR forced to stay between 0 and 1, 

'''
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
'''

def total_LBF_analisys(models,phishing_URLs,X_train,y_train,X_test,y_test,testing_list,device,verbose):
  save_Backup(models,phishing_URLs,X_train,y_train,X_test,y_test,testing_list,device,verbose)
  LBF_graph(models,phishing_URLs,X_train,y_train,device,True,True,True,verbose)

'''
- rimossi alcuni import
- messi parametri globali
- aggiunto if else per evitare di ripetere analisi di tau quando i file sono giá presenti
'''