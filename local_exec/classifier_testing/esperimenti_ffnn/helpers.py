from os.path import exists
import numpy as np
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_extraction.text import CountVectorizer

def load_data(verbose = True):
    '''
    Carica ed effettua prepocessing del dataset di legitimate e phishing URL.
    I file del dataset devono trovarsi all'interno della cartella indicata dal parametro glocale loc_data ed avere estensione .npy
    '''
    
    legitimate_URLs = np.load("C:/Users/Giacomo/Desktop/Università/Tesi/Codice/lbf-tirocinio/data/dataset1/legitimate_URLs.npy")
    phishing_URLs = np.load("C:/Users/Giacomo/Desktop/Università/Tesi/Codice/lbf-tirocinio/data/dataset1/phishing_URLs.npy")
    phishing_URLs = np.concatenate((phishing_URLs,np.load("C:/Users/Giacomo/Desktop/Università/Tesi/Codice/lbf-tirocinio/data/dataset2/phishing_URLs2.npy", allow_pickle=True)))
    legitimate_URLs = np.concatenate((legitimate_URLs,np.load("C:/Users/Giacomo/Desktop/Università/Tesi/Codice/lbf-tirocinio/data/dataset2/legitimate_URLs2.npy", allow_pickle=True)))
    legitimate_URLs = np.concatenate((legitimate_URLs,np.load("C:/Users/Giacomo/Desktop/Università/Tesi/Codice/lbf-tirocinio/data/dataset3/legitimate_URLs3.npy", allow_pickle=True)))
    
     # clean URLs
    legitimate_URLs = [l.split('http://')[-1].split('www.')[-1].split('https://')[-1] for l in legitimate_URLs]
    phishing_URLs = [p.split('http://')[-1].split('www.')[-1].split('https://')[-1] for p in phishing_URLs]

    # Rimuovo duplicati
    phishing_URLs = list(set(phishing_URLs) - set(legitimate_URLs))
    legitimate_URLs = list(set(legitimate_URLs))
    phishing_URLs.sort()
    legitimate_URLs.sort()

    # randomly permute URLs
    phishing_URLs = np.array(phishing_URLs)
    legitimate_URLs = np.array(legitimate_URLs)
    np.random.seed(seed=0)
    legitimate_URLs = list(legitimate_URLs[np.random.permutation(len(legitimate_URLs))])
    phishing_URLs = list(phishing_URLs[np.random.permutation(len(phishing_URLs))])
    print("legit:",legitimate_URLs[:10])  

    if verbose:
        print(len(legitimate_URLs),len(phishing_URLs), len(legitimate_URLs) + len(phishing_URLs))

    X = np.asarray(phishing_URLs + legitimate_URLs)
    y = np.asarray([1 for i in range(len(phishing_URLs))] + [0 for i in range(len(legitimate_URLs))])

    return X, y

def encode(X, y):
    """ Codifica il dataset in ingresso sfruttando CountVectorizer, y non viene modificato """

    vectorizer = CountVectorizer(analyzer = 'char') 
    X_encoded = vectorizer.fit_transform(X)

    return X_encoded.toarray(), y, vectorizer.vocabulary_

def rnn_map_to_number(X, y):
    letters = ''.join(X) #String unica di URL senza spazi
    c = Counter(letters) # Counter con occorrenze delle lettere
    d = {}

    for i, (l, _) in enumerate(c.most_common(128)):
        d[l] = i + 1 # Dizionario ordinato per numero di occorrenze ( associa ad ogni lettera il suo rank)

    return d

def rnn_encode(X, y, dict, char_cutoff = 150):
    X_encoded = [[dict[l] if l in dict else 0 for l in url] for url in X]
    X_encoded = [[l for l in url[:min([len(url),char_cutoff])]] + [0 for l in range(char_cutoff-len(url))] for url in X_encoded]
    y_encoded = y

    return np.array(X_encoded), np.array(y_encoded)

def undersample(X, y, ratio = None):
    """ Ribilancia il dataset in ingresso in base alla proporzione indicata in ratio """
    
    if ratio is not None:
        undersample = RandomUnderSampler(sampling_strategy = ratio)
        X_under, y_under = undersample.fit_resample(X, y)
        return X_under, y_under

    return X, y

def shuffle(X, y, seed = 0):
    """ Ritorna una permutazione di X,y generata con il seed in input"""

    np.random.seed(seed)
    perm = np.random.permutation(len(X))

    X = X[perm]
    y = y[perm]

    return X, y