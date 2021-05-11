
from sklearn.model_selection import train_test_split, GridSearchCV , StratifiedKFold,KFold, RandomizedSearchCV
import numpy as np
import pandas as pd
import math
import init
from classificatori.helpers import cutoff,makedf,get_set_stratified, codificate
from classificatori.train_and_val import train_and_fit
from scipy import stats
savepath = "~/latex/Model_selection_"

def my_Grid_search(estimator, params, nmparams, multilevel, X_train,y_train):
    gridmodel = GridSearchCV(estimator, params, scoring='f1_macro', cv=5, return_train_score=True)          #creo oggetto GridSearch
    gridmodel.fit(X_train, y_train)                                      #effettuo ricerca su X_trainval
    if(multilevel):
        value = gridmodel.best_params_[nmparams]
        newparams = grid_params(math.log10(value)-1, math.log10(value)+1, name=nmparams, nelem=len(params[nmparams]))
        gridmodel= GridSearchCV(estimator, newparams, scoring='f1_macro', cv=5, return_train_score=True) 
        gridmodel.fit(X_train, y_train)
    print(gridmodel.best_params_)                                             #stampo best param
    return gridmodel

def my_randomyze(estimator, params, X_train,y_train, iter =5):
    randomodel = RandomizedSearchCV(estimator, params,n_iter=iter,n_jobs=-1)
    randomodel.fit(X_train,y_train)
    print(randomodel.best_params_)                                             #stampo best param
    return randomodel


def ModelSelection(codifica, estimator, params, name, nmparams, Randomize = False, lcutoff = None, multilevel = False):
    #divisone partizioni
    X, y, kf = get_set_stratified(lcutoff)
    rlist={'accuracy':[],
           'f1-score':[]}
    svname = ""
    for nmpar in nmparams:
        rlist[nmpar]:[]
    
    for train,test in kf.split(X,y):
        X_test, X_train = X[test], X[train]
        y_test, y_train = y[test], y[train]
        X_train, X_test, svname = codificate(X_train, X_test, codifica, svname)
        if Randomize:
            model = my_randomyze(estimator, params, X_train,y_train)
        else:
            model = my_Grid_search(estimator, params, nmparams, multilevel, X_train,y_train)
        res = train_and_fit(X_train, y_train, X_test, y_test, model.best_estimator_, name, False)
        for nmpar in nmparams:
            rlist[nmpar].append(model.best_params_[nmpar])
        rlist['accuracy'].append(res['accuracy'])
        rlist['f1-score'].append(res['Phishing']['f1-score'])
    dfparam = pd.DataFrame(rlist)
    print(dfparam)
    dfparam.to_latex(buf=savepath + name + svname + "model_selection_params.tex")
    metrics=['f1-score','accuracy']
    makedf(rlist, metrics, savepath + name + svname + "model_selection_score.tex")

    


def grid_params(start, end,name,nelem=None,logspace = True):
    if (logspace):
        if(nelem==None):
            nelem=end-start+1
        print(np.logspace(start, end, nelem))
        return {name : np.logspace(start, end, nelem)}
    if(nelem==None):
        nelem = int(end/start)
    return {name : np.linspace(start,end,nelem)}


def randomize_params(scaleC=100,scaleGamma=0.1):
    return {'C' : stats.expon(scaleC),'gamma': stats.expon(scale=scaleGamma) }
