from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer,  TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV , StratifiedKFold
from sklearn.linear_model import SGDClassifier, LogisticRegression
import sklearn
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import init


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

def Bayes_clasifier(alpha_params=1):       
    return MultinomialNB(alpha=alpha_params)
#definisco all'inizio griglia val. fissi parametri; per classificatore dizionario che associa a params fissi valori fissi
def SVM_classifier(C_param=1.0, penality_param='l2', loss_param = 'squared_hinge'):
    return sklearn.svm.LinearSVC(C=C_param, penalty=penality_param, loss=loss_param, max_iter=10000)
    
def LogisticRegression_classifier(C_param=1, max_iter_param=2000, solver_param='lbfgs'):
    return LogisticRegression(C=C_param, max_iter=max_iter_param, solver=solver_param)

def get_set(codif=None,cutoff=None):
        X, y = init.load_data() #prendo i dati (argomento uno va ad indicare il ratio fra phishing e legitimate
                                                        #ho scelto di usare un dataset bilanciato)
        X =X.tolist()
        if(cutoff):
            for i,x in enumerate(X):
                X[i]= x[:min(len(x),cutoff)]
            pX = []
            py = []
            x_prev = ""
            xy =  [(x,y) for x,y in zip(X,y)]
            for x,y in sorted(xy,key=lambda x:x[0]):                #ciclo per eliminare elementi doppi (X) e relative y dopo il cutoff
                if(x!=x_prev):
                    pX.append(x)
                    py.append(y)
                x_prev=x
            X = pX
            y = py
        X_train,  X_test,  y_train,  y_test =train_test_split(X, y, test_size=0.3)    #divido Train e test set
        assert(len(X_train) == len(list(set(X_train)-set(X_test))))             #assertion error in caso di elementi presenti sia in X_train che in X_test
       
        if(codif==None):                                                        #default  ritorna X_train, y_train...
            return X_train, y_train, X_test, y_test

        codif.fit(X)                                                            #creo dizionario con le parole
        X_train= codif.transform(X_train)                                       #applico codifica (bag of word/bag of char) su dataset per intero
        X_test = codif.transform(X_test)
        return X_train, y_train, X_test, y_test                                    #ritorna X_train, X_test codificati

def makedf(list, metrics, names, iter= False):                                    #stampa DF con risultati ed salva in latex
    diz= {}
    if (iter == True):
        for name in names:
            for metric in metrics:
                diz[metric] ={
                    'min': np.amin(list[name][metric]), 
                    'max':np.amax(list[name][metric]), 
                    'avg':np.average(list[name][metric]), 
                    'dev_std':np.std(list[name][metric])
                }
            dframe = pd.DataFrame(diz)
            dframe.to_latex(buf=f"/home/dav/Scrivania/latex/{name}char.tex") 
            print(pd.DataFrame(diz))
    else:
        for name in names:
            for metric in metrics:
                diz[metric] = list[name][metric]
            dframe = pd.DataFrame(diz)
            dframe.to_latex(buf=f"/home/dav/Scrivania/latex/{name}char.tex")
            print(pd.DataFrame(diz))

def  analysis(iteration, namelist, classifiers, codif, verbose=False, ts=None):     #funzione per andare a effettuare una o più iterazioni dati i classificatori
    metrics = ['precision', 'recall', 'f1-score', 'accuracy']
    rlist = {}
    for name in namelist:
        rlist[name]= {}
        for metric in metrics:
            rlist[name][metric] = []

    for _ in range(iteration):
        if(ts == None):
            X_train,  y_train,  X_test,  y_test = get_set(codif)
        else:
            X_train, y_train, X_test, y_test= ts
        for classifier, name in zip(classifiers, namelist):
            res=train_and_fit(X_train, y_train, X_test, y_test, classifier, name, verbose)
            for metric in metrics:
                if(metric == 'accuracy'):
                    rlist[name][metric]+= [(res[metric])]
                else:  
                    rlist[name][metric]+= [res['Phishing'][metric]]

    makedf(rlist, metrics, namelist, (iteration>1))

def GridModelSelection(codifica, estimator, params, name, nmparams):
    #divisone partizioni
    X,y = init.load_data(False)
    X = np.array(X)
    y = np.array(y)
    codifica.fit(X)
    kf=StratifiedKFold(n_splits=5)
    rlist={ nmparams: [],
           'accuracy':[],
           'f1-score':[]}
    for train,test in kf.split(X,y):
        X_test, X_train = X[test], X[train]
        y_test, y_train = y[test], y[train]
        assert(len(X_test)==len(list(set(X_test)-set(X_train))))
        X_train = codifica.transform(X_train)
        X_test = codifica.transform(X_test)
        gridmodel = GridSearchCV(estimator, params, scoring='f1_macro', cv=5, return_train_score=True)          #creo oggetto GridSearch
        gridmodel.fit(X_train, y_train)                                      #effettuo ricerca su X_trainval
        print(gridmodel.best_params_)                                             #stampo best param
        res = train_and_fit(X_train, y_train, X_test, y_test, gridmodel.best_estimator_, name, False)
        rlist[nmparams] += [gridmodel.best_params_[nmparams]]
        rlist['accuracy'] += [res['accuracy']]
        rlist['f1-score'] += [res['Phishing']['f1-score']]
    dfparam = pd.DataFrame(rlist)
    metrics=['f1-score','accuracy']
    valutazione = {}
    for metric in metrics:
        valutazione[metric] = {'avg' : np.average(rlist[metric]),
                                'std': np.std(rlist[metric])}

    dfval = pd.DataFrame(valutazione)
    dfparam.to_latex(buf=f"/home/dav/Scrivania/latex/{name}charparams-score.tex")
    dfval.to_latex(buf=f"/home/dav/Scrivania/latex/{name}charvalidation.tex")
    print(dfparam)
    print(dfval)
  

def select_params(start, end,name,nelem=None,logspace = True):
    if (logspace):
        if(nelem==None):
            nelem=end-start+1
        print(np.logspace(start, end, nelem))
        return {name : np.logspace(start, end, nelem)}
    if(nelem==None):
        nelem = int(end/start)
    return {name : np.linspace(start,end,nelem)}


if __name__ == "__main__":
    codif =  CountVectorizer(analyzer='char')
    namelist = ["Naive Bayes", "SVM", "Logistic regression"]
    classifiers = [Bayes_clasifier(), SVM_classifier(), LogisticRegression_classifier()]
    GridModelSelection(codif, Bayes_clasifier(), select_params(-10,10,"alpha"), "Naive bayes","alpha")
    #GridModelSelection(codif, LogisticRegression_classifier(), select_params(0,6,'C'), "Logistic Regression","C")
    #GridModelSelection(codif, SVM_classifier(), select_params(-3, 1), "Linear SVM","C")
    #analysis(10, namelist,classifiers, codif,verbose=False) 
       
