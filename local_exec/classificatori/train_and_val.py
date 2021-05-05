from sklearn.naive_bayes import MultinomialNB
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
import helpers
import init

savepath = "/home/dav/Scrivania/latex/" + "analysis"
metrics = ['precision', 'recall', 'f1-score', 'accuracy']

def train_and_fit(X_train, y_train, X_test, y_test, classifier, name, verbose=True):  #classe per fare training e valutare il classificatore  -> train and set
    nb = Pipeline([
                ('tfidf',  TfidfTransformer()),                   #tfidf,  va ad assegnare un vettore dei pesi in base alla composizione del testo 
                ('clf',  classifier),                             #applica classificatore passato come parametro
                ])
    nb.fit(X_train,  y_train)                                    #tfidf su X_train e training del classificatore
    y_pred = nb.predict(X_test)                                 #tfidf su X_test e calcollo di y pred
    if(verbose):
        print(f"{name}:")
        print('accuracy %s' % sklearn.metrics.accuracy_score(y_test, y_pred))
        print(classification_report(y_test,  y_pred))
    return classification_report(y_test, y_pred, output_dict=True,target_names=["Legitimate","Phishing"])    # return classification report

def  analysis_holdout(iteration, classifier_name, classifiers, codif, verbose=False, ts=None,cutoff=None,frel=False,pca=False,small=False):     #funzione per andare a effettuare una o più iterazioni dati i classificatori
    lsa = None
    get_ts = None
    X,y=init.load_data()
    if(small):
        X,_,y,_ = train_test_split(X,y,train_size=0.1)    #considero solo un decimo del dataset causalmente mantenendo la stratificazione
        get_ts = (X,y)
    codif.fit(X)  
    if(pca):               
        if(paramcodif=='word'):
            lsa = TruncatedSVD(n_components=nfeatureword)
            #lsa = PCA(n_components=0.80)
        else:
            lsa = TruncatedSVD(n_components=nfeaturechar)

        lsa.fit(codif.transform(X))
        print(lsa.n_components)
        print("varianza informazione: ",np.sum(lsa.explained_variance_ratio_))
    rlist = {}
    for name in classifier_name:
        rlist[name]= {}
        for metric in metrics:
            rlist[name][metric] = []

    for _ in range(iteration):
        if(ts == None):
            X_train,  y_train,  X_test,  y_test = get_set(codif,cutoff,frel,lsa,get_ts)
        else:
            X_train, y_train, X_test, y_test= ts
        for classifier, name in zip(classifiers, classifier_name):
            res=train_and_fit(X_train, y_train, X_test, y_test, classifier, name, verbose)
            for metric in metrics:
                if(metric == 'accuracy'):
                    rlist[name][metric]+= [(res[metric])]
                else:  
                    rlist[name][metric]+= [res['Phishing'][metric]]

    makedf(rlist, metrics, classifier_name,(iteration>1))

def Cross_Validation_analisys(classifier_list,classifier_name,codif,componenti=None,frel=False,verbose=False):
    X,y = init.load_data(False)
    X = np.array(X)
    y = np.array(y)
    kf = StratifiedKFold()
    svname = ""
    rlist = {}
    for name in classifier_name:
        rlist[name]= {}
        for metric in metrics:
            rlist[name][metric] = []
    
    for train,test in kf.split(X,y):    
        X_test, X_train = X[test], X[train]
        y_test, y_train = y[test], y[train]
        assert(len(X_test)==len(list(set(X_test)-set(X_train))))
        codif.fit(X_test)
        if(frel):
            X_train = helpers.toRel(codif.transform(X_train),X_train)
            X_test = helpers.toRel(codif.transform(X_test),X_test)
            svname+="_lrel"
        else:
            X_train = codif.transform(X_train)
            X_test = codif.transform(X_test)
            
        if(componenti!=None):
            lsa = TruncatedSVD(n_components=componenti)
            X_train,X_test = helpers.fit_lsa(X_train,X_test,lsa)
            svname+=f"_sva{componenti}"
            print("varianza informazione: ",np.sum(lsa.explained_variance_ratio_))


        for classifier, name in zip(classifier_list, classifier_name):
            res=train_and_fit(X_train, y_train, X_test, y_test, classifier, name, verbose)
            for metric in metrics:
                if(metric == 'accuracy'):
                    rlist[name][metric]+= [(res[metric])]
                else:  
                    rlist[name][metric]+= [res['Phishing'][metric]]

    helpers.makedf(rlist, metrics, classifier_name ,savepath+"_CRV"+svname)
