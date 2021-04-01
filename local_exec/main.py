# Libraries
import trainRNN #parametri per training della RNN -> per info train.py
import analysisBF
import torch.nn as nn
import analysisLBF
import analysisSLBF
import init
import analysisTau
#interfaces

def load_RNN():
    return trainRNN.load_eval(X_train,y_train,criterion,h_sizes,emb_size,batch_size)
def train():
    trainRNN.train(X_train,y_train,criterion,h_sizes,emb_size,batch_size)
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
    analysisLBF.total_LBF_analisys(models,phishing_URLs,X_train,y_train,X_test,y_test,testing_list,verbose)
#SLBF analysis
def SLBF_tau_analysis(verbose = True):
    helpers.tau_analysis(models, phishing, X_train, y_train, ("false_negs2",  "taus2"))
def SLBF_Bloom_filters_analysis(verbose=True): #crea Bloom filters necessari per SLBF e li salva in TrainNN/analysis
    analysisSLBF.SLBF_Bloom_filters(models,phishing,X_train,y_train,X_test,y_test,testing_list, ("false_negs2",  "taus2"), verbose=True)
def SLBF_graph(falseN=True,FPR=True,Size=True,verbose=False): #vedi LBF_graph
    analysisSLBF.SLBF_graph(models,phishing,X_train,y_train, ("false_negs2",  "taus2"))
def SLBF_total_analisys(verbose=True):
    analysisSLBF.SLBF_total_analisys(models,phishing_URLs,X_train,y_train,X_test,y_test,testing_list,verbose)

#esecuzione
rate_lp,h_sizes,emb_size,batch_size = init.get_arguments()
print(f"il ratio legit/phish è uguale a {rate_lp} , hsizes sono uguali a {h_sizes} mentre emb_size è uguale a {emb_size}")
criterion = nn.CrossEntropyLoss()  #imposto criterion (utilizzato per la funzione di loss in fase di training e di valutazione)
legitimate_URLs,phishing_URLs = init.load_data(rate_lp) #first pre processing on input
print(len(legitimate_URLs),len(phishing_URLs))
X_train,y_train,X_test,y_test,training_list,testing_list = init.training_set(legitimate_URLs,phishing_URLs)  #return training set; see init.py for details
train() #eseguire solo la prima volta, parametri inseriti in config.
models = load_RNN()
#analysis function
LBF_total_analisys(False)
SLBF_total_analisys(False)

'''
import non aggioranti
parametri vecchi
)
'''