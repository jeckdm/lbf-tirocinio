import LBF
import config 
import numpy as np
loc = config.loc_nn
fprs = config.fprs
def tau_analysis(models, phishing_URLs,X_train,y_train, name,slbf, verbose = True):
	'''
	calcola la tau basandosi su predefiniti fpr e fprs-rate e se verbose=true stampa a video i risultati dell'analisi.
	i falsi negativi e le tau relativi ad ogni (fpr, fprs-rate) sono salvati come name[0].npy e name[1].npy in loc e vengono poi ritornati dalla funzione
	'''
	fpr_ratios=config.fpr_ratios
	if slbf :
		fpr_ratios = config.fpr_ratios2

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


'''
    -aggiunta per probelmi reccurent import
	-aggiunto argomento per indicare se uso fpr_ratios o fprs_ratios2

'''