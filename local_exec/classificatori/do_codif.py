from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import  TfidfTransformer
import numpy as np
from scipy import sparse
import re
def cutoff(X, y, length):
    for i,x in enumerate(X):
        X[i] = x[:min(len(x), length)]
    pX = []
    py = []
    x_prev = ""
    xy =  [(x, y1) for x, y1 in zip(X, y)]
    for x,y in sorted(xy, key=lambda x:x[0]):    #ciclo per eliminare elementi doppi (X) e relative y dopo il cutoff
        if(x != x_prev):
            pX.append(x)
            py.append(y)
        x_prev = x
    return pX, py

def fit_lsa(X_train,X_test,lsa):
    lsa.fit(X_train)
    X_train = lsa.transform(X_train)
    X_test = lsa.transform(X_test)
    print("varianza informazione: ", np.sum(lsa.explained_variance_ratio_))
    return X_train,X_test


def toRel(X_codif,X):
    X_len = [len(x) for x in X]
    X_codif = X_codif.toarray()
    for i, x in enumerate(X_codif):
        if X_len[i] == 0:
            X_len[i] = 1
        X_codif[i] = x * 10000 / X_len[i] 
    return sparse.csr_matrix(X_codif)

def codificate (X_train, X_test, codif, path, frel=False, nfeatures = None, esperimenti = False, pre_codif = False):
    lsa = None
    if(not pre_codif):
        codif.fit(X_train)
    if(frel):
        X_train = toRel(codif.transform(X_train),X_train)
        X_test = toRel(codif.transform(X_test),X_test)
        path=path+'_frel'
    else:
        X_train = codif.transform(X_train)
        if esperimenti:
            esperimenti_codifica(X_test, codif)
        X_test = codif.transform(X_test)

    if(nfeatures!=None):
        lsa = TruncatedSVD(n_components = nfeatures, random_state = 42)
        X_train,X_test = fit_lsa(X_train, X_test,lsa)
        path+=f"_sva{nfeatures}" 
        

    tdf = TfidfTransformer()
    tdf.fit(X_train)
    tdf.transform(X_train)
    tdf.transform(X_test)
    return X_train, X_test, path ,(tdf,lsa)

def esperimenti_codifica(X_test,codif):
    lista = codif.get_feature_names()
    tot_count = 0
    u_count = 0
    is_in = False
    w_count = 0
    for url in X_test:
        url = url.lower()
        words = re.split(r"\b", url)
        for word in words:
            if re.search(r'\w',word)!=None:
                w_count+=1
                if  word not in lista:
                    tot_count +=1 
                    is_in=True
        if is_in == True:
            u_count+=1
            is_in = False

    print(f" parole totali: {w_count} parole non presenti: {tot_count}, url_totali : {len(X_test)} url influenzati : {u_count}, ")