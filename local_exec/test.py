import BF
import RNN
import trainRNN
import torch.nn as nn 
import torch
import init
import analysisBF
import analysisLBF
import analysisSLBF

# Parametri globali
import config

device = config.device

# Setting location
# config.loc_data = ...
# config.loc_nn = ...
# config.loc_plots = ...

# Loading dataset e suddivisione in training/testing
legitimate, phishing = init.load_data()
X_train,y_train,X_test,y_test,training_list,testing_list = init.get_train_set(legitimate, phishing)

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

# Analisi e grafici
analysisLBF.save_Backup(models,phishing,X_train,y_train,X_test,y_test,testing_list,verbose=True)
analysisLBF.LBF_graph(models,phishing,X_train,y_train)