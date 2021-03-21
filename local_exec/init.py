from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
#platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag()) variable never used
# cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/' must run in env
import torch
import numpy as np
def GPU_init():                           #if NVIDIA then accelerator else none, da rivedere, mette sempre cpu
    if exists('/dev/nvidia0'):
        device = torch.device('CUDA') 
        accelerator = cuda_output[0]
    else:
        device = torch.device('cpu') 
        accelerator= 'cpu' 
    return device,accelerator

def take_input():
    # Download data
    legitimate_URLs = np.load("small_data/legitimate_URLs.npy")
    phishing_URLs = np.load("small_data/phishing_URLs.npy")
    #phishing_URLs = np.concatenate((phishing_URLs,np.load("small_data/phishing_URLs2.npy",allow_pickle=True)))
    #legitimate_URLs=np.concatenate((legitimate_URLs,np.load("small_data/legitimate_URLs2.npy",allow_pickle=True)))

    # randomly permute URLs
    np.random.seed(0)
    legitimate_URLs = list(legitimate_URLs[np.random.permutation(len(legitimate_URLs))])
    phishing_URLs = list(phishing_URLs[np.random.permutation(len(phishing_URLs))])
    # clean URLs
    legitimate_URLs = [l.split('http://')[-1].split('www.')[-1].split('https://')[-1] for l in legitimate_URLs]
    phishing_URLs = [p.split('http://')[-1].split('www.')[-1].split('https://')[-1] for p in phishing_URLs]
    phishing_URLs = list(set(phishing_URLs) - set(legitimate_URLs))
    return legitimate_URLs,phishing_URLs

def map_to_number(legitimate_URLs,phishing_URLs):
    letters = ''.join(legitimate_URLs+phishing_URLs) #String unica di URL senza spazi
    from collections import Counter
    c = Counter(letters) # Counter con occorrenze delle lettere
    d = {}
    print(len(c))
    for i, (l, _) in enumerate(c.most_common(128)):
        if(i>128):
            break
        d[l] = i + 1 # Dizionario ordinato per numero di occorrenze ( associa ad ogni lettera il suo rank)
    return d

def training_set(d,legitimate_URLs,phishing_URLs):
    # training set: all keys plus non-keys
    # testing set: remaining non-keys 
    training_list = legitimate_URLs[:int(len(legitimate_URLs)/2)]+phishing_URLs
    testing_list = legitimate_URLs[int(len(legitimate_URLs)/2):]
        
    # cut off at 150 chars
    char_cutoff = 150

    # Matrice con entry i vari URL del training set portati a 150 char con eventuale padding/troncatura.
    # Gli URL vengono rappresentati sostiuendo ad ogni lettera l'int rappresentante la posizione nelle occorrenze
    X_train = torch.tensor([[d.get(l,0) for l in url[:min([len(url),char_cutoff])]]+[0 for l in range(char_cutoff-len(url))] 
                            for url in training_list])
    y_train = torch.tensor([0]*int(len(legitimate_URLs)/2)+[1]*int(len(phishing_URLs)))

    X_test = torch.tensor([[d.get(l,0) for l in url[:min([len(url),char_cutoff])]]+[0 for l in range(char_cutoff-len(url))] 
                            for url in testing_list])
    y_test = torch.tensor([0]*(len(testing_list)))

    return X_train,y_train,X_test,y_test,training_list,testing_list

def get_train_set(legitimate_URLs,phishing_URLs):
    return training_set(map_to_number(legitimate_URLs,phishing_URLs),legitimate_URLs,phishing_URLs)  # interfaccia utente, applica le fasi di pre-processing e ritorna i traing set
