from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split , StratifiedKFold
from sklearn.linear_model import LogisticRegression
import sklearn
import numpy as np
import pandas as pd
import init
from classificatori import do_codif


def SVM_linear_classifier(C_param=1.0, penality_param='l2', loss_param = 'squared_hinge'):
    return sklearn.svm.LinearSVC(C=C_param, penalty=penality_param, loss=loss_param,max_iter=3000)
def SVM_classifier(kernel='rbf',gamma='scale'):
    return sklearn.svm.SVC(kernel=kernel,gamma = gamma )
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

def get_set_stratified(lcutoff=None, small = False):
    X, y = init.load_data(False)
    if lcutoff:
        X, y = do_codif.cutoff(X, y, lcutoff)
    if small:
        X,_,y,_ = train_test_split(X,y,train_size=0.1,random_state=42)

    X = np.array(X)
    y = np.array(y)
    kf = StratifiedKFold()
    return X, y, kf

def get_set_holdout(codif=None, lcutoff=None, frel=False,lsa=None,ts=None, small = True):
        X, y = init.load_data() #prendo i dati (argomento uno va ad indicare il ratio fra phishing e legitimate        
        if small:
            X,y,_,_ = train_test_split(X,y,train_size=0.1,random_state=42)                                            
        if(ts!=None):
            X,y = ts
        if(lcutoff!=None):                
            X, y = do_codif(X, y, lcutoff)
            #savepath = savepath + "cutoff" + lcutoff
        
        X_train,  X_test,  y_train,  y_test = train_test_split(X, y, test_size=0.3)    #divido Train e test set
        assert(len(X_train) == len(list(set(X_train) - set(X_test))))             #assertion error in caso di elementi presenti sia in X_train che in X_test
        if(codif == None):                                                        #default  ritorna X_train, y_train...
            return X_train, y_train, X_test, y_test

        codif.fit(X_test)
        cX_train = codif.transform(X_train)                                       #applico codifica (bag of word/bag of char) su dataset per intero
        cX_test = codif.transform(X_test)
        if(frel):
            cX_train = do_codif.toRel(cX_train, X_train)
            cX_test = do_codif.toRel(cX_test, X_test)        
        if(lsa!=None):
            cX_train,cX_test = do_codif.fit_lsa(cX_train,cX_test)
        return cX_train, y_train, cX_test, y_test           
                                 #ritorna X_train, X_test codificati

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
                'avg':np.average(list[metric]), 
                'dev_std':np.std(list[metric])
            }
        to_df(diz, savepath)


def to_df(diz,savepath,name=None):
    dframe = pd.DataFrame(diz)
    print(pd.DataFrame(diz))
    if name:
        savepath = savepath + name
    dframe.to_latex(buf = savepath)