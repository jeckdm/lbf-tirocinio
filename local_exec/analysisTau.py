import LBF
import config 
import numpy as np
import pandas as pd
import glob
import os

loc = config.loc_nn

'''
calcola la tau basandosi su predefiniti fpr e fprs-rate e se verbose=true stampa a video i risultati dell'analisi.
i falsi negativi e le tau relativi ad ogni (fpr, fprs-rate) sono salvati come name[0].npy e name[1].npy in loc e vengono poi ritornati dalla funzione
Falsi negativi che vengono calcolati sulla base dei risultati del classificatore models sulla lista di phishing URLs in ingresso (Tramite relativa tau)
'''
def tau_analysis(models, fprs, fpr_ratios, training_list, X_train, y_train, name = ("false_negs", "taus"), verbose = True):


	false_negs = {}
	taus = {}
	# Per ognuno dei modelli salvo numero di falsi negativi del classificatore e tau ottimale nelle relative strutture sulla
	# base del fprs e fpr_ratio target.
	for i in range(len(models)): # Cambiato range da 3 a models
		false_negs[i] = {}
		taus[i] = {}
		for fpr in fprs:
			for fpr_ratio in fpr_ratios:
				false_negs[i][(fpr,fpr_ratio)], taus[i][(fpr,fpr_ratio)] = LBF.build_LBF_classifier(models[i], fpr*fpr_ratio, X_train, y_train, training_list)
				if(verbose):
					print("Modello %d: fpr: %.3f, fpr ratio: %.2f, FNR: %.20f, %.10f" % (i, fpr, fpr_ratio, len(false_negs[i][(fpr,fpr_ratio)])/len(training_list), taus[i][(fpr, fpr_ratio)]))

	np.save(loc+name[0], false_negs)
	np.save(loc+name[1], taus)

	return false_negs,taus

'''
'''
def get_models_size():
  # Ritorna grandezza dei modelli nell'ordine in cui sono salvati nella cartella
  sizes = []

  for filename in list(glob.glob(config.loc_nn + "RNN_*")):
    sizes.append(os.path.getsize(filename))

  print(sorted(sizes))

  return sorted(sizes)

'''
'''
def fnrs_analysis(models, fprs, fpr_ratios, false_negs, training_list):
  # Per ognuno dei modelli costruisco un dataframe in cui salvo il rate di falsi negativi per ogni fpr e fprs_ratio
  fnrs = {}

  for i in range(len(models)): # Cambiato range da 3 a models
    fnrs[i] = pd.DataFrame(index=fpr_ratios, columns=fprs)
    for fpr in fprs:
      for fpr_ratio in fpr_ratios:
        fnrs[i].loc[fpr_ratio,fpr] = len(false_negs[i][(fpr,fpr_ratio)])/len(training_list)

  return fnrs

'''
    -aggiunta per probelmi reccurent import
	-aggiunto argomento per indicare se uso fpr_ratios o fprs_ratios2

'''