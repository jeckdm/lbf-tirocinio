from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import SGDClassifier,LogisticRegression
import sklearn
import numpy as np
import pandas as pd
import init


def generic_classifier(X_train,y_train,X_test,y_test,classifier,name,verbose=True):
    nb = Pipeline([
                ('tfidf', TfidfTransformer()),
                ('clf', classifier),
                ])
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    if(verbose):
        print(f"{name}:")
        print('accuracy %s' % sklearn.metrics.accuracy_score(y_test,y_pred))
        print(classification_report(y_test, y_pred))
    return classification_report(y_test,y_pred,output_dict=True)

def Bayes_clasifier(alpha_params=1):
    return MultinomialNB(alpha=alpha_params)

def SVM_classifier(C_param=1.0,penality_param='l2',loss_param = 'squared_hinge'):
    return sklearn.svm.LinearSVC(C=C_param,penalty=penality_param,loss=loss_param)
    
def LogisticRegression_classifier(C_param=1,max_iter_param=2000,solver_param='lbfgs'):
    return LogisticRegression(C=C_param,max_iter=max_iter_param,solver=solver_param)

def get_set(sbilanciamento,codif=None):
        legitimate_URLs,phishing_URLs = init.load_data(sbilanciamento) #prendo i dati (argomento uno va ad indicare il ratio fra phishing e legitimate
                                                        #ho scelto di usare un dataset bilanciato)
        X= legitimate_URLs+phishing_URLs
        y= [0]*len(legitimate_URLs)+ [1]*len(phishing_URLs)      #associo ad ogni legit 0 e ad ogni phish 1
        X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.3)
        if(codif==None):
            return X_train,y_train,X_test,y_test
        codif.fit(X)
        X_train= codif.transform(X_train)
        X_test = codif.transform(X_test)
        return X_train,y_train,X_test,y_test
def makedict(list,metrics,names,iter= False):
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
            dframe.to_latex(buf=f"/home/dav/Scrivania/latex/{name}.tex") 
            print(pd.DataFrame(diz))
    else:
        for name in names:
            for metric in metrics:
                diz[metric] = list[name][metric]
            dframe = pd.DataFrame(diz)
            dframe.to_latex(buf=f"/home/dav/Scrivania/latex/{name}es1.tex")
            print(pd.DataFrame(diz))

def  analysis(iteration,namelist,classifiers,codif,verbose=False,sbilanciamento=1,ts=None):
    metrics = ['precision','recall','f1-score','accuracy']
    rlist = {}
    for name in namelist:
        rlist[name]= {}
        for metric in metrics:
            rlist[name][metric] = []

    for i in range(iteration):
        if(ts == None):
            X_train,y_train,X_test,y_test=get_set(sbilanciamento,codif)
        else:
            X_train,y_train,X_test,y_test=ts
        for j,classifier,name in zip(range(0,len(classifiers)),classifiers,namelist):
            res=generic_classifier(X_train,y_train,X_test,y_test,classifier,name,False)
            for k,metric in zip(range(0,len(metrics)),metrics):
                if(metric == 'accuracy'):
                    rlist[name][metric]+=[(res[metric])]
                else:  
                    rlist[name][metric]+=[res['weighted avg'][metric]]
    makedict(rlist,metrics,namelist,(iteration>1))

def GridModelSelection(sbilanciamento,codifica,estimator,params,name):
    X_train,y_train,X_test,y_test =get_set(sbilanciamento)
    X_trainval,X_validation,y_trainval,y_validation= train_test_split(X_train,y_train,test_size=0.3)
    print(len(X_train+y_train))
    codifica.fit(X_train+X_test)
    X_trainval=codifica.transform(X_trainval)
    X_validation = codifica.transform(X_validation)
    X_train = codifica.transform(X_train)
    X_test = codifica.transform (X_test)
    gridmodel = GridSearchCV(estimator,params,scoring='f1_weighted')
    gridmodel.fit(X_trainval,y_trainval)
    print(gridmodel.best_params_)
    analysis(1,[name],[gridmodel.best_estimator_],codifica,ts=(X_train,y_train,X_test,y_test))


def Bayesparams(start,end,granularity):
    print(np.arange(start,end,granularity))
    return {'alpha' : np.arange(start,end,granularity)}

codif =  CountVectorizer(analyzer='word')
namelist = ["Naive Bayes","SVM","Logistic regression"]
classifiers = [Bayes_clasifier(),SVM_classifier(),LogisticRegression_classifier()]

GridModelSelection(1,codif,Bayes_clasifier(),Bayesparams(0,0.2,0.00001),"Naive Bayes")
