from os.path import exists
import torch
import numpy as np
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler

# Parametri globali
import config

# Setting device
device = config.device

def load_data(verbose = True):
    '''
    Carica ed effettua prepocessing del dataset di legitimate e phishing URL.
    I file del dataset devono trovarsi all'interno della cartella indicata dal parametro glocale loc_data ed avere estensione .npy
    '''
    
    legitimate_URLs = np.load(config.loc_data1 + "legitimate_URLs.npy")
    phishing_URLs = np.load(config.loc_data1 + "phishing_URLs.npy")
    phishing_URLs = np.concatenate((phishing_URLs,np.load(config.loc_data2 + "phishing_URLs2.npy",allow_pickle=True)))
    legitimate_URLs = np.concatenate((legitimate_URLs,np.load(config.loc_data2 + "legitimate_URLs2.npy",allow_pickle=True)))
    legitimate_URLs = np.concatenate((legitimate_URLs,np.load(config.loc_data3 + "legitimate_URLs3.npy",allow_pickle=True)))

    # randomly permute URLs
    np.random.seed()
    legitimate_URLs = list(legitimate_URLs[np.random.permutation(len(legitimate_URLs))])
    phishing_URLs = list(phishing_URLs[np.random.permutation(len(phishing_URLs))])

    # clean URLs
    legitimate_URLs = [l.split('http://')[-1].split('www.')[-1].split('https://')[-1] for l in legitimate_URLs]
    phishing_URLs = [p.split('http://')[-1].split('www.')[-1].split('https://')[-1] for p in phishing_URLs]

    # Rimuovo duplicati
    phishing_URLs = list(set(phishing_URLs) - set(legitimate_URLs))
    legitimate_URLs = list(set(legitimate_URLs))

    if verbose:
        print(len(legitimate_URLs),len(phishing_URLs), len(legitimate_URLs) + len(phishing_URLs))

    X = np.asarray(phishing_URLs + legitimate_URLs)
    y = np.asarray([1 for i in range(len(phishing_URLs))] + [0 for i in range(len(legitimate_URLs))])

    return X, y
    
def map_to_number(X, y, char_cutoff = 150):
    '''
    Ritorna dataset codificati: assegna ad ogni carattere dell'URL un intero univoco in base al numero di
    occorrenze, URL piú frequenti hanno un numero piú basso.
    Il dataset risultante é un tensore
    '''

    letters = ''.join(X) #String unica di URL senza spazi
    c = Counter(letters) # Counter con occorrenze delle lettere
    d = {}

    for i, (l, _) in enumerate(c.most_common(128)):
        d[l] = i + 1 # Dizionario ordinato per numero di occorrenze ( associa ad ogni lettera il suo rank)
    
    X_encoded = [[d[l] if l in d else 0 for l in url] for url in X]
    X_encoded = torch.tensor([[l for l in url[:min([len(url),char_cutoff])]] + [0 for l in range(char_cutoff-len(url))] for url in X_encoded])
    y_encoded = torch.tensor(y)

    return X_encoded, y_encoded

def undersample(X, y, ratio = None):
    if ratio is not None:
        undersample = RandomUnderSampler(sampling_strategy = ratio)
        X_under, y_under = undersample.fit_resample(X.reshape(-1,1), y)
        return X_under.reshape(-1), y_under

    return X, y

def LBF_train_test_split(X, y, X_encoded):
    '''
    Ritorna dataset in ingresso suddivisi in training (legit/2 + phishing) e testing set (legit/2) con relativi output.
    '''

    legitimate_URLs = X[[y[i] == 0 for i in range(len(X))]]
    phishing_URLs = X[[y[i] == 1 for i in range(len(X))]]

    legitimate_URLs_encoded = X_encoded[[y[i] == 0 for i in range(len(X_encoded))]]
    phishing_URLs_encoded = X_encoded[[y[i] == 1 for i in range(len(X_encoded))]]

    # training set: all keys plus non-keys
    # testing set: remaining non-keys 
    training_list = np.array(phishing_URLs)
    testing_list = legitimate_URLs[int(len(legitimate_URLs)/2):]
        
    X_train = torch.cat((legitimate_URLs_encoded[:int(len(legitimate_URLs_encoded)/2)],phishing_URLs_encoded))
    y_train = torch.tensor([0]*int(len(legitimate_URLs_encoded)/2)+[1]*int(len(phishing_URLs_encoded)))

    X_test = legitimate_URLs_encoded[int(len(legitimate_URLs_encoded)/2):]
    y_test = torch.tensor([0]*(len(testing_list)))

    return X_train, y_train, X_test, y_test, training_list, testing_list
