from sklearn.naive_bayes import MultinomialNB
import sys
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer,  TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV , StratifiedKFold,KFold
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA,TruncatedSVD,IncrementalPCA
import sklearn
from scipy import sparse
import numpy as np
import pandas as pd
import math
import init


def SVM_linear_classifier(C_param=1.0, penality_param='l2', loss_param = 'squared_hinge'):
    return sklearn.svm.LinearSVC(C=C_param, penalty=penality_param, loss=loss_param,max_iter=3000)
def SVM_classifier(kernel='rbf',gamma='scale'):
    return sklearn.svm.SVC()
def LogisticRegression_classifier(C_param=1, solver_param='lbfgs'):
    return LogisticRegression(C=C_param, solver=solver_param,max_iter=3000)


def get_default_classifier_list(Bayes=False,Linear_SVM=False,Logistic= False, SVM = False):
    classifier_list = []
    name_list = []
    if(Bayes):
        classifier_list.append(MultinomialNB())
        name_list.append("Bayes")
    if(Linear_SVM):
        classifier_list.append(SVM_linear_classifier())
        name_list.append("Linear SVM")
    if(Logistic):
        classifier_list.append(LogisticRegression_classifier())
        name_list.append("Logistic Regression")
    if(SVM):
        classifier_list.append(SVM_classifier())
        name_list.append("SVM")
    return name_list,classifier_list

    

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

def get_set_stratified(lcutoff=None):
    X, y = init.load_data(False)
    if(lcutoff):
        X, y = cutoff(X, y, lcutoff)
    X = np.array(X)
    y = np.array(y)
    kf = StratifiedKFold()
    return X, y, kf

def get_set_holdout(codif=None, lcutoff=None, frel=False,lsa=None,ts=None):
        X, y = init.load_data() #prendo i dati (argomento uno va ad indicare il ratio fra phishing e legitimate                                                    
        if(ts!=None):
            X,y = ts
        if(lcutoff!=None):                
            X, y = cutoff(X, y, lcutoff)
            savepath = savepath + "cutoff" + lcutoff
        
        X_train,  X_test,  y_train,  y_test = train_test_split(X, y, test_size=0.3)    #divido Train e test set
        assert(len(X_train) == len(list(set(X_train) - set(X_test))))             #assertion error in caso di elementi presenti sia in X_train che in X_test
        if(codif == None):                                                        #default  ritorna X_train, y_train...
            return X_train, y_train, X_test, y_test

        codif.fit(X_test)
        cX_train = codif.transform(X_train)                                       #applico codifica (bag of word/bag of char) su dataset per intero
        cX_test = codif.transform(X_test)
        if(frel):
            cX_train = toRel(cX_train, X_train)
            cX_test = toRel(cX_test, X_test)        
        if(lsa!=None):
            cX_train,cX_test = fit_lsa(cX_train,cX_test)
        return cX_train, y_train, cX_test, y_test           
                                 #ritorna X_train, X_test codificati


def fit_lsa(X_train,X_test,lsa):
    lsa.fit(X_train)
    X_train = lsa.transform(X_train)
    X_test = lsa.transform(X_test)
    return X_train,X_test


def toRel(X_codif,X):
    X_len = [len(x) for x in X]
    X_codif = X_codif.toarray()
    for i, x in enumerate(X_codif):
        if X_Len[i] == 0:
            X_Len[i] = 1
        X_codif[i] = x * 10000 / X_Len[i] 
    return sparse.csr_matrix(X_codif)


def makedf(list, metrics, savepath,names = None, iter=True):                                    #stampa DF con risultati ed salva in latex
    diz= {}
    if names:
        for name in names:
            if(iter):
                for metric in metrics:
                    diz[metric] ={
                        'avg':np.average(list[name][metric]), 
                        'dev_std':np.std(list[name][metric])
                    }
            else:
                for metric in metrics:
                    diz[metric] = list[name][metric]
            to_df(diz, savepath,name)
    else:
        for metric in metrics:
            diz[metric] = {
                'avg':np.average(list[name][metric]), 
                'dev_std':np.std(list[name][metric])
            }
        to_df(diz, savepath)


def to_df(diz,savepath,name=None):
    dframe = pd.DataFrame(diz)
    print(pd.DataFrame(diz))
    if name:
        savepath = savepath + name
    dframe.to_latex(buf = savepath) 


def codificate (X_train,X_test,codif,path,frel=None,lsa = None):
    codif.fit(X_train)
    assert(len(X_test)==len(list(set(X_test)-set(X_train))))
    if(frel):
        X_train = toRel(codif.transform(X_train),X_train)
        X_test = toRel(codif.transform(X_test),X_test)
        path=path+'_frel'
    else:
        X_train = codifica.transform(X_train)
        X_test = codifica.transform(X_test)
    if(lsa!=None):
        lsa = TruncatedSVD(n_components=componenti,random_state=42)
        X_train,X_test = helpers.fit_lsa(X_train,X_test,lsa)
        path+=f"_sva{componenti}"
        print("varianza informazione: ",np.sum(lsa.explained_variance_ratio_))
    return X_train,X_test,path
        
