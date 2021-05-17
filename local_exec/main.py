# Libraries
import trainRNN #parametri per training della RNN -> per info train.py
import analysisBF
import torch.nn as nn
import torch
import analysisLBF
import analysisSLBF
import init
import analysisTau
import graph
import argparse
import RNN
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter

#interfaces
def load_RNN():
    return trainRNN.load_eval(X_test, y_test, criterion, h_sizes ,emb_size, batch_size)
def train():
    trainRNN.train(X_train, y_train, criterion, h_sizes, emb_size, batch_size)
#BF
def BF_test_size():
    analysis.BF_test_size(phishing_URLs,testing_list)
#LBF analysis
def LBF_tau_analysis(verbose=True):#calcola la tau basandosi su predefiniti fpr e fprs-rate e se verbose=true stampa a video i risultati dell'analisi.
    analysisTau.tau_analysis(models, phishing, X_train, y_train, ("false_negs",  "taus")) # Analisi tau e salvataggio false_negs e taus per ogni (fpr, fprs_ratio)
def save_Backup(verbose=True): #salva filtri bloom di backup in trained_NN/analisys; se verbose = true stampa fpr emirico, size e tempo
    analysisLBF.save_Backup(models,phishing,X_train,y_train,X_test,y_test,testing_list, ("false_negs",  "taus"),verbose=True) # Salvataggio BF backup  per ogni (fpr_ fprs_ratio)
def LBF_graph(falseN=True,FPR=True,Size=True,verbose=False): #stampa grafici, specifico quali grafici (default tutti) e se stampare msg (default true). richiede esecuzione precedente di save Backup
    analysisLBF.LBF_graph(models,phishing,X_train,y_train, ("false_negs","taus"),False,falseN,FPR,Size,verbose) # Generazioni grafici
def LBF_total_analisys(verbose = True): #fa analisi completa del lbf, va a eseguire internamente tau analisis, elabora e salva i backup filter e crea i grafici
    analysisLBF.total_LBF_analisys(models, fprs, fpr_ratios, LBF_training_list, LBF_X_train, LBF_y_train, LBF_X_test, LBF_y_test, LBF_testing_list)
#SLBF analysis
def SLBF_tau_analysis(verbose = True):
    helpers.tau_analysis(models, phishing, X_train, y_train, ("false_negs2",  "taus2"))
def SLBF_Bloom_filters_analysis(verbose=True): #crea Bloom filters necessari per SLBF e li salva in TrainNN/analysis
    analysisSLBF.SLBF_Bloom_filters(models,phishing,X_train,y_train,X_test,y_test,testing_list, ("false_negs2",  "taus2"), verbose=True)
def SLBF_graph(falseN=True,FPR=True,Size=True,verbose=False): #vedi LBF_graph
    analysisSLBF.SLBF_graph(models,phishing,X_train,y_train, ("false_negs2",  "taus2"))
def SLBF_total_analisys(verbose=True):
    analysisSLBF.SLBF_total_analisys(models, fprs, fpr_ratios+fpr_ratios2, LBF_training_list, LBF_X_train, LBF_y_train, LBF_X_test, LBF_y_test, LBF_testing_list)

def main(args):
    # Setting parametri ricevuti da linea di comando
    rate_lp = args.ratiolp
    h_sizes=args.hsizes
    batch_size= args.batchsize
    emb_size = args.embsize    

    fprs = [0.001,0.005,0.01,0.02]
    fpr_ratios = [0.1*i for i in range(1,11)] # Ratio per LBF
    fpr_ratios2 = [1.*i for i in range(1,11)] # Ratio per SLBF

    print(f"il ratio legit/phish è uguale a {rate_lp} , hsizes sono uguali a {h_sizes} mentre emb_size è uguale a {emb_size}")

    #Carico dataset
    X, y = init.load_data() #first pre processing on input
    print(f"Dataset caricato, Numero Legit(0) e Phishing(1): {Counter(y)}, Grandezza totale Dataset: {len(X)}")
    X, y = init.undersample(X, y, ratio = args.ratiolp) # Undersampling del dataset
    print(f"Dataset undersampled con ratio = {rate_lp}, Risultato: Numero Legit(0) e Phishing(1): {Counter(y)}, Grandezza totale Dataset: {len(X)}")
    X_encoded, y_encoded = init.map_to_number(X, y) # Codifico gli URL, risultato numpy array
    print("Url codificati")

    # Suddivisione dataset per training e valutazione del classificatore
    X_train, y_train, X_test, y_test = train_test_split(X_encoded, y_encoded) # Suddivisione in training e testing
    print(f"Dataset splittato in training e testing per il classificatore con rapporto 0.75, Risultato: Grandezza train {np.shape(X_train)}, {np.shape(y_train)}, Grandezza test: {np.shape(X_test)}, {np.shape(y_test)}")

    criterion = nn.CrossEntropyLoss()  #imposto criterion (utilizzato per la funzione di loss in fase di training e di valutazione)

    # Suddivisione dataset per creazione ed analisi delle strutture
    LBF_X_train, LBF_y_train, LBF_X_test, LBF_y_test, LBF_training_list, LBF_testing_list = init.LBF_train_test_split(X, y, X_encoded)  # Suddivisione in training e testing set (tensor) + list (np array), al momento sono tensor posso cambiarli?
    print (f"X_train: {len(LBF_X_train)} (Dovrebbe essere {310329/2 + 43744 }), X_test: {len(LBF_X_test)} (Dovrebbe essere {310329/2})")
    print(np.shape(LBF_training_list))

    # Training classificatore + analisi
    trainRNN.train(torch.tensor(X_encoded), torch.tensor(y_encoded), criterion, h_sizes, emb_size, batch_size) #eseguire solo la prima volta, parametri inseriti in config.
    models = trainRNN.load_eval(LBF_X_test, LBF_y_test, criterion, h_sizes ,emb_size, batch_size)
    print(models)
    
    #analysis function
    analysisLBF.total_LBF_analisys(models, fprs, fpr_ratios, LBF_training_list, LBF_X_train, LBF_y_train, LBF_X_test, LBF_y_test, LBF_testing_list)
    analysisSLBF.SLBF_total_analisys(models, fprs, fpr_ratios+fpr_ratios2, LBF_training_list, LBF_X_train, LBF_y_train, LBF_X_test, LBF_y_test, LBF_testing_list)
    graph.classifier_graph(models, X_train, y_train, "classifier_distribution")

if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Argument for analysis")
    parse.add_argument("--ratiolp","-r",help="specify the ratio legit/phishing",type=float,default = None)
    parse.add_argument("--embsize","-e",help="specify the size of the emb",type=int,default=5 )
    parse.add_argument("--batchsize","-b",help="specify the batch size", type=int,default=256)
    parse.add_argument("--hsizes","-s",help="specify the sizes of the GRU",type=int,nargs='*', default=[16,8,4])

    args=parse.parse_args()

    main(args)
