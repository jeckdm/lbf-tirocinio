import BF
import RNN
import trainRNN
import torch.nn as nn 
import torch
import init
import analysisBF
import analysisLBF
import analysisSLBF
import helpers

# Parametri globali
import config

device = config.device

# Setting location
config.loc_data = 'C:/Users/Giacomo/Desktop/Università/Tesi/Codice/lbf-tirocinio/small_data/'
# config.loc_nn = ...
# config.loc_plots = ...

# Loading dataset e suddivisione in training/testing
legitimate, phishing = init.load_data()
X_train, y_train, X_test, y_test, training_list, testing_list = init.get_train_set(legitimate, phishing)

print(f"Training X: {X_train.shape}, Training y: {y_train.shape}")
print(f"Testing X: {X_test.shape}, Testing y: {y_test.shape}")

# Test
FPR, BF_size, t = BF.run_BF(0.02, phishing, testing_list)
print("FPR", FPR, "size", BF_size, "time", t)

# Funzione di loss
criterion = nn.CrossEntropyLoss()

# Volendo setto i parametri config.h_sizes = [...] ... oppure uso quelli di default di config.py


# Addestramento del modello (Questa sarebbe la parte da cambiare nel caso volessimo cambiare classificatore)
for i in range(4):
    models = {}

    # Create model, loss function, optimizer
    models[i] = RNN.RNN(emb_size=config.emb_size, h_size=config.h_sizes[i], layers=config.layers).to(device)
    optimizer = torch.optim.Adamax(models[i].parameters())

    trainRNN.train(models[i], X_train, y_train, optimizer, criterion)
    torch.save(models[i].state_dict(), config.loc_nn+"RNN_emb"+str(config.emb_size)+"_hid"+str(config.h_sizes[i]))  

# Test loading dei modelli
models = trainRNN.load_eval(X_test, y_test, criterion)


# Eventuale setting dei parametri per analisi dei grafici config.fpr, config.fprs_raio... 


# Analisi e grafici

# L'idea per una analisi completa (file + grafici) é di fare LBF_tau_analysis, save_Backup per i file, LBF_graph per generare i grafi 
# (Forse posso togliere chiamata e relativo controllo a LBF_tau_analysis in LBF_graph assumendo che i file debbano giá essere presenti come accade per LBF_saveBackup?)

config.fpr_ratios = [0.1*i for i in range(1,11)]

helpers.tau_analysis(models, phishing, X_train, y_train, ("false_negs",  "taus")) # Analisi tau e salvataggio false_negs e taus per ogni (fpr, fprs_ratio)
analysisLBF.save_Backup(models,phishing,X_train,y_train,X_test,y_test,testing_list, ("false_negs",  "taus"),verbose=True) # Salvataggio BF backup  per ogni (fpr_ fprs_ratio)
analysisLBF.LBF_graph(models,phishing,X_train,y_train, ("false_negs",  "taus")) # Generazioni grafici

# Stesso ragionamento per SLBF
config.fpr_ratios = config.fpr_ratios + [1.*i for i in range(1,11)]

helpers.tau_analysis(models, phishing, X_train, y_train, ("false_negs2",  "taus2")) # Ripeto analisi di tau aumentando peró il numero di fpr_ratio (Potrei aggiungere un parametro per settare il nome del file per non sovrascriverlo?)
analysisSLBF.SLBF_Bloom_filters(models,phishing,X_train,y_train,X_test,y_test,testing_list, ("false_negs2",  "taus2"), verbose=True)
analysisSLBF.SLBF_graph(models,phishing,X_train,y_train, ("false_negs2",  "taus2"))