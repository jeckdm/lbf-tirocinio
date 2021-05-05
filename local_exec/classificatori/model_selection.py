
from sklearn.model_selection import train_test_split, GridSearchCV , StratifiedKFold,KFold
import numpy as np
import pandas as pd
import math
import init
from classificatori.helpers import cutoff
from classificatori.train_and_val import train_and_fit
savepath = "/home/dav/Scrivania/latex/Model_selection_"

def GridModelSelection(codifica, estimator, params, name, nmparams,lcutoff=None,multilevel=False):
    #divisone partizioni
    X,y = init.load_data(False)
    if(lcutoff!=None):
        X,y=cutoff(X,y,lcutoff)
    X = np.array(X)
    y = np.array(y)
    kf=StratifiedKFold(n_splits=5)
    rlist={ nmparams: [],
           'accuracy':[],
           'f1-score':[]}
    for train,test in kf.split(X,y):
        gridmodel = GridSearchCV(estimator, params, scoring='f1_macro', cv=5, return_train_score=True,verbose=3)          #creo oggetto GridSearch
        X_test, X_train = X[test], X[train]
        y_test, y_train = y[test], y[train]
        codifica.fit(X_train)
        assert(len(X_test)==len(list(set(X_test)-set(X_train))))
        X_train = codifica.transform(X_train)
        X_test = codifica.transform(X_test)
        gridmodel.fit(X_train, y_train)                                      #effettuo ricerca su X_trainval
        if(multilevel):
            value = gridmodel.best_params_[nmparams]
            newparams = select_params(math.log10(value)-1, math.log10(value)+1, name=nmparams, nelem=len(params[nmparams]))
            gridmodel= GridSearchCV(estimator, newparams, scoring='f1_macro', cv=5, return_train_score=True) 
            gridmodel.fit(X_train, y_train)
        print(gridmodel.best_params_)                                             #stampo best param
        res = train_and_fit(X_train, y_train, X_test, y_test, gridmodel.best_estimator_, name, False)
        rlist[nmparams].append(gridmodel.best_params_[nmparams])
        rlist['accuracy'].append(res['accuracy'])
        rlist['f1-score'].append(res['Phishing']['f1-score'])
    dfparam = pd.DataFrame(rlist)
    metrics=['f1-score','accuracy']
    valutazione = {}
    for metric in metrics:
        valutazione[metric] = {'avg' : np.average(rlist[metric]),
                                'std': np.std(rlist[metric])}

    dfval = pd.DataFrame(valutazione)
    dfparam.to_latex(buf=savepath + name + "model_selection_params.tex")
    dfval.to_latex(buf=savepath + name + "model_selection_score.tex")
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

