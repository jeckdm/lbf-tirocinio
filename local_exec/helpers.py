import math
import numpy as np
import LBF

# Parametri globali
import config 

fprs = config.fprs
fpr_ratios = config.fpr_ratios
loc = config.loc_nn

# helper functions to do the math 

# given over-all FPR and  F_p, F_n of f, compute bits per element for BF
def solve_LBF_size(FPR, F_p, F_n, alpha=0.6185):
	ratio  = (FPR-F_p) / (1-F_p)
	b = F_n * math.log(ratio) / math.log(alpha)
	return b

def solve_SBF_size(FPR, F_p, F_n, alpha=0.6185):
	ratio = F_p / ((1-F_p)*(1/F_n-1))
	b_2 = F_n *  math.log(ratio) / math.log(alpha)
	constant = F_p + (1-F_p)* (alpha ** (b_2 / F_n))
	b_1 = math.log(FPR/constant) / math.log(alpha)
	return  b_1, b_2

def solve_FPR_LBF(b, F_p, F_n, alpha=0.6185):
	return  F_p+(1-F_p)* (alpha ** (b / F_n))

def solve_FPR_SBF(b1, b2, F_p, F_n, alpha=0.6185):
	return  (alpha ** b1)* (F_p + (1-F_p)* (alpha  ** (b2 / F_n)))

def determine_tau(FPR_tau, prob_list):
  return np.percentile(np.array(prob_list),100*(1.-FPR_tau))

def tau_analysis(models, phishing_URLs,X_train,y_train, name, verbose = True):
	'''
	calcola la tau basandosi su predefiniti fpr e fprs-rate e se verbose=true stampa a video i risultati dell'analisi.
	i falsi negativi e le tau relativi ad ogni (fpr, fprs-rate) sono salvati come name[0].npy e name[1].npy in loc e vengono poi ritornati dalla funzione
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
					print("Modello %d: fpr: %.3f, fpr ratio: %.2f, FNR: %.20f, %.10f" % (i, fpr, fpr_ratio, len(false_negs[i][(fpr,fpr_ratio)])/len(phishing_URLs), taus[i][(fpr, fpr_ratio)]))

	np.save(loc+name[0], false_negs)
	np.save(loc+name[1], taus)

	return false_negs,taus