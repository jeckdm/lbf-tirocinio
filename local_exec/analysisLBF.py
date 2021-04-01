import BF
import LBF
import pandas as pd
import numpy as np
import helpers
import os.path
import analysisTau
import glob
import graph
from os import path

# Parametri globali
import config

device = config.device

'''
Salva filtri bloom di backup in trained_NN/anlisys; se verbose = true stampa fpr empirico, size e tempo dei filtri di backup creati

Il parametro name contiene la coppia che raprresenta i nomi dei file da cui vengono caricati il il dizionario (fpr, fpr_ratio): falsi negativi e (fpr, fpr_ratio): tau
se tali file non esistono vengono creati con il nome indicato in name tramite una chiamata a helpers.tau_analysis
'''
def create_BFsbackup(models, fprs, fpr_ratios, false_negs):
  LBF_backups = {}

  # Per ognuno dei modelli salvo il filtro di backup costruito sulla base del fpr e fpr_ratio target
  for i in range(len(models)): 
    LBF_backups[i] = {}
<<<<<<< HEAD
    for fpr in config.fprs:
        for fpr_ratio in config.fpr_ratios:
          try:
            LBF_backups[i][(fpr,fpr_ratio)] = LBF.build_LBF_backup(false_negs[i][(fpr,fpr_ratio)], fpr, fpr*fpr_ratio)
            if(LBF_backups[i][(fpr,fpr_ratio)] =='error'):
                continue
            fpr0, BF_size, t = LBF.test_LBF(models[i], LBF_backups[i][(fpr,fpr_ratio)], taus[i][(fpr,fpr_ratio)],X_test,y_test,testing_list)
            if(verbose):
              print(f"teoric fpr: {fpr}, empirc fpr: {fpr0}, size of backup BF: {BF_size}, time : {t}")
            model_size =  os.path.getsize(config.loc_nn+"RNN_emb"+str(config.emb_size)+"_hid"+str(config.h_sizes[i])) # Calcolo size classificatore
            if(verbose):
              print("SIZE MODELLO: ",  model_size)
            LBFo = {"FPR": fpr0, "size": BF_size+model_size, "time": t}
            np.save(config.loc_nn+"LBF_hid"+str(config.h_sizes[i])+"_FPR"+str(fpr)+"_ratio"+str(fpr_ratio), LBFo)
          except ZeroDivisionError:
              # Se il numero di falsi negativi é 0 sollevo eccezione e non salvo
              # Non controllata inizialmente probabilmente perché é stata esclusa la possibilitá di avere fn = 0 con dataset grandi
              if(verbose):
                print("Numero falsi negativi = 0")
=======
>>>>>>> 4295ea0245d41b912e2acf2260354db11e0b0c7c

    for fpr in fprs:
      for fpr_ratio in fpr_ratios:
        try:
          LBF_backups[i][(fpr,fpr_ratio)] = LBF.build_LBF_backup(false_negs[i][(fpr,fpr_ratio)], fpr, fpr*fpr_ratio)
          if(LBF_backups[i][(fpr,fpr_ratio)] =='error'):
            continue
        except ZeroDivisionError:
            # Se il numero di falsi negativi é 0 sollevo eccezione e non salvo
            print("Numero falsi negativi = 0")

  return LBF_backups

'''
'''
def empirical_analysis(models, fprs, fpr_ratios, LBFs, X_test, y_test, testing_list, taus, save = False):
  # Per ogni modello salvo in base a fpr, fpr_ratio target l'frp empirico, la grandezza ed il tempo di accesso per elemento del LBF relativa
  # Utile per i grafici successivi
  true_fpr_LBF = {}
  sizes_LBF = {}
  times_LBF = {}

  # Grandezze dei modelli addestrati (Dei soli parametri (?))
  models_size = analysisTau.get_models_size()

  for i in range(len(models)): # Cambiato range da 3 a models
    true_fpr_LBF[i] = pd.DataFrame(index = fpr_ratios, columns = fprs)
    sizes_LBF[i] = pd.DataFrame(index = fpr_ratios, columns = fprs)
    times_LBF[i] = pd.DataFrame(index = fpr_ratios, columns = fprs)

    for fpr in fprs:
      for fpr_ratio in fpr_ratios:
        try:
          # Carico il relativo SLBF/LBF
          BF_backup = LBFs[i][(fpr,fpr_ratio)]
          # Calcolo parametri empirici
          fpr0, BF_size, t = LBF.test_LBF(models[i], BF_backup, taus[i][(fpr,fpr_ratio)],X_test,y_test,testing_list)
          # Calcolo la size del modello
          model_size = models_size[i]
          # Salvo i risultati
          true_fpr_LBF[i].loc[fpr_ratio,fpr] = fpr0
          sizes_LBF[i].loc[fpr_ratio,fpr] = BF_size + model_size
          times_LBF[i].loc[fpr_ratio,fpr] = t
          print(f"FPR Target, FPR Ratio: ({fpr},{fpr_ratio}), FPR empirico: {fpr0}, Size totale: {BF_size + model_size}, Tempo di accesso medio: {t}")
          # Genero i file
          if save:
            LBFo = {"FPR": fpr0, "size": BF_size+model_size, "time": t}
            # Andrebbe cambiato formato di salvataggio
            # np.save(config.loc_nn+"LBF_hid"+str(config.h_sizes[i])+"_FPR"+str(fpr)+"_ratio"+str(fpr_ratio), LBFo)
        except:
          # Aggiunta except utile nel caso in cui il file non fosse stato salvato perché fn = 0
          print("error / numero falsi negativi 0")
          continue

  return true_fpr_LBF, sizes_LBF, times_LBF

def total_LBF_analisys(models, fprs, fpr_ratios, training_list, X_train, y_train, X_test, y_test, testing_list, verbose=False):
  # Faccio analisi di tau e salvo relativi file
  print("ANALISI TAU")
  false_negs, taus = analysisTau.tau_analysis(models, fprs, fpr_ratios, training_list, X_train, y_train, name=("false_negs", "taus"))
  # false_negs = np.load(config.loc_nn + "false_negs.npy", allow_pickle=True).item()
  # taus = np.load(config.loc_nn + "taus.npy", allow_pickle=True).item()
  # Creo i filtri di backup sulla base di fprs, fpr_ratios
  print("CREAZIONI BF BACKUP")
  LBF_backups = create_BFsbackup(models, fprs, fpr_ratios, false_negs)
  # Calcolo rate di falsi negativi per ogni fprs, fpr_ratios
  print("CALCOLO FNRS")
  fnrs = analysisTau.fnrs_analysis(models, fprs, fpr_ratios, false_negs, training_list)
  # Analisi empirica delle strutture create
  print("ANALISI EMPIRICA")
  true_fpr_LBF, sizes_LBF, times_LBF = empirical_analysis(models, fprs, fpr_ratios, LBF_backups, X_test, y_test, testing_list, taus)
  # Genero grafici
  graph.LBF_graph(fnrs, true_fpr_LBF, sizes_LBF, "LBF")

'''
- rimossi alcuni import
- messi parametri globali
- aggiunto if else per evitare di ripetere analisi di tau quando i file sono giá presenti
'''