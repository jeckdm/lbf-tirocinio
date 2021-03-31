from os.path import exists
from wheel.pep425tags import get_abbr_impl, get_impl_ver, get_abi_tag
#platform = '{}{}-{}'.format(get_abbr_impl(), get_impl_ver(), get_abi_tag()) variable never used
# cuda_output = !ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/' must run in env
import torch
import numpy as np
from collections import Counter
import argparse
# Parametri globali
import config

device = config.device
def get_arguments():
    parse = argparse.ArgumentParser(description="Argument for analysis")
    parse.add_argument("--ratiolp","-r",help="specify the ratio legit/phishing",type=int,default=-1)
    parse.add_argument("--embsize","-e",help="specify the size of the emb",type=int,default=5 )
    parse.add_argument("--batchsize","-b",help="specify the batch size", type=int,default=256)
    parse.add_argument("--hsizes","-s",help="specify the sizes of the GRU",type=int,nargs='*', default=[4,8,16])
    args=parse.parse_args()
    ratelp = args.ratiolp
    h_sizes=args.hsizes
    batch_size= args.batchsize
    emb_size = args.embsize
    return ratelp,h_sizes,emb_size,batch_size

def load_data(rate_lp):
    '''
    Carica ed effettua prepocessing del dataset di legitimate e phishing URL.
    I file del dataset devono trovarsi all'interno della cartella indicata dal parametro glocale loc_data ed avere estensione .npy
    '''
    
    legitimate_URLs = np.load(config.loc_data1 + "legitimate_URLs.npy")
    phishing_URLs = np.load(config.loc_data1 + "phishing_URLs.npy")
    phishing_URLs = np.concatenate((phishing_URLs,np.load(config.loc_data2 + "phishing_URLs2.npy",allow_pickle=True)))
    legitimate_URLs=np.concatenate((legitimate_URLs,np.load(config.loc_data2 + "legitimate_URLs2.npy",allow_pickle=True)))
    legitimate_URLs=np.concatenate((legitimate_URLs,np.load(config.loc_data3 + "legitimate_URLs3.npy",allow_pickle=True)))

    # randomly permute URLs
    np.random.seed(0)
    legitimate_URLs = list(legitimate_URLs[np.random.permutation(len(legitimate_URLs))])
    phishing_URLs = list(phishing_URLs[np.random.permutation(len(phishing_URLs))])

    # clean URLs
    legitimate_URLs = [l.split('http://')[-1].split('www.')[-1].split('https://')[-1] for l in legitimate_URLs]
    phishing_URLs = [p.split('http://')[-1].split('www.')[-1].split('https://')[-1] for p in phishing_URLs]
    phishing_URLs = list(set(phishing_URLs) - set(legitimate_URLs))
    legitimate_URLs = list(set(legitimate_URLs))
    print(len(legitimate_URLs),len(phishing_URLs))
    if(rate_lp == -1):   #nel caso il ratio non sia specificato (-1) torno i due dataset originari
        return legitimate_URLs,phishing_URLs
    if(rate_lp*len(phishing_URLs)<=len(legitimate_URLs)):#nel caso ratio*phishing <= legit diminuisco il numero di legit in maniera da rendere
        return legitimate_URLs[:rate_lp*len(phishing_URLs)],phishing_URLs #vero il ratio e ritorno questo e il dataset phishing originario    
    else:
        diff = int(legitimate_URLs/ratio)  #trovo la lenght dei phish per rendere valido il ratio
        return legitimate_URLs,phishing_URLs[:diff]
def map_to_number(legitimate_URLs,phishing_URLs):
    '''
    Ritorna dataset codificati: assegna ad ogni carattere dell'URL un intero univoco in base al numero di
    occorrenze, URL piú frequenti hanno un numero piú basso.
    '''

    letters = ''.join(legitimate_URLs+phishing_URLs) #String unica di URL senza spazi
    c = Counter(letters) # Counter con occorrenze delle lettere
    d = {}
    for i, (l, _) in enumerate(c.most_common(128)):
        d[l] = i + 1 # Dizionario ordinato per numero di occorrenze ( associa ad ogni lettera il suo rank)
    
    legitimate_URLs = [[d[l] if l in d else 0 for l in url] for url in legitimate_URLs]
    phishing_URLs = [[d[l] if l in d else 0 for l in url] for url in phishing_URLs]

    return legitimate_URLs, phishing_URLs

def training_set(legitimate_URLs,phishing_URLs):
    '''
    Ritorna dataset in ingresso suddivisi in training (legit/2 + phishing) e testing set (legit/2) con relativi output.
    '''

    # training set: all keys plus non-keys
    # testing set: remaining non-keys 
    training_list = legitimate_URLs[:int(len(legitimate_URLs)/2)]+phishing_URLs
    testing_list = legitimate_URLs[int(len(legitimate_URLs)/2):]
        
    # URL codificati
    train, test = map_to_number(training_list,testing_list)

    # cut off at 150 chars
    char_cutoff = 150

    # Matrice con entry i vari URL del training set portati a 150 char con eventuale padding/troncatura.
    # Gli URL vengono rappresentati sostiuendo ad ogni lettera l'int rappresentante la posizione nelle occorrenze
    X_train = torch.tensor([[l for l in url[:min([len(url),char_cutoff])]] + [0 for l in range(char_cutoff-len(url))] for url in train])
    y_train = torch.tensor([0]*int(len(legitimate_URLs)/2)+[1]*int(len(phishing_URLs)))

    X_test = torch.tensor([[l for l in url[:min([len(url),char_cutoff])]] + [0 for l in range(char_cutoff-len(url))] for url in test])
    y_test = torch.tensor([0]*(len(testing_list)))

    return X_train,y_train,X_test,y_test,training_list,testing_list
'''
def get_train_set(legitimate_URLs,phishing_URLs):

    Codifica i dataset in input e li ritorna suddivisi in training e testing. 
    
    return training_set(legitimate_URLs, phishing_URLs)  # interfaccia utente, applica le fasi di pre-processing e ritorna i traing set
    
    lo toglierei dato che va sempliemente a chiamare training set
''' 
'''
- sistemato (?) cuda device
- map_to_number ritorna i dataset codificati, non il dizionario,
  codifica viene fatta in training_set subito dopo suddivisione in training e testing list
'''
