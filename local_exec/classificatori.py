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


def generic_classifier(X_train,y_train,X_test,y_test,classifier,name,verbose=True):  #classe per fare training e valutare il classificatore
    nb = Pipeline([
                ('tfidf', TfidfTransformer()),                  #tfidf, va ad assegnare un vettore dei pesi in base alla composizione del testo 
                ('clf', classifier),                            #applica classificatore passato come parametro
                ])
    nb.fit(X_train, y_train)                                    #tfidf su X_train e training del classificatore
    y_pred = nb.predict(X_test)                                 #tfidf su X_test e calcollo di y pred
    if(verbose):
        print(f"{name}:")
        print('accuracy %s' % sklearn.metrics.accuracy_score(y_test,y_pred))
        print(classification_report(y_test, y_pred))
    return classification_report(y_test,y_pred,output_dict=True)    # return classification report

def Bayes_clasifier(alpha_params=1):       
    return MultinomialNB(alpha=alpha_params)

def SVM_classifier(C_param=1.0,penality_param='l2',loss_param = 'squared_hinge'):
    return sklearn.svm.LinearSVC(C=C_param,penalty=penality_param,loss=loss_param,max_iter=10000)
    
def LogisticRegression_classifier(C_param=1,max_iter_param=2000,solver_param='lbfgs'):
    return LogisticRegression(C=C_param,max_iter=max_iter_param,solver=solver_param)

def get_set(sbilanciamento,codif=None):
        X,y = init.load_data(sbilanciamento) #prendo i dati (argomento uno va ad indicare il ratio fra phishing e legitimate
                                                        #ho scelto di usare un dataset bilanciato)
        X =X.tolist()
        y=y.tolist()
        X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.3)    #divido Train e test set
        if(len(X_train) != len(list(set(X_train)-set(X_test)))):                #assertion error in caso di elementi presenti sia in X_train che in X_test
            raise AssertionError()
        if(codif==None):                                                        #default  ritorna X_train,y_train...
            return X_train,y_train,X_test,y_test
        codif.fit(X)                                                            #creo dizionario con le parole
        X_train= codif.transform(X_train)                                       #applico codifica (bag of word/bag of char) su dataset per intero
        X_test = codif.transform(X_test)
        return X_train,y_train,X_test,y_test                                    #ritorna X_train,X_test codificati

def makedf(list,metrics,names,iter= False):                                    #stampa DF con risultati ed salva in latex
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

def  analysis(iteration,namelist,classifiers,codif,verbose=False,sbilanciamento=1,ts=None):     #funzione per andare a effettuare una o più iterazioni dati i classificatori
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
    makedf(rlist,metrics,namelist,(iteration>1))

def GridModelSelection(sbilanciamento,codifica,estimator,params,name):
    X_train,y_train,X_test,y_test =get_set(sbilanciamento)
    X_trainval,X_validation,y_trainval,y_validation= train_test_split(X_train,y_train,test_size=0.3)
    codifica.fit(X_train+X_test)
    X_trainval=codifica.transform(X_trainval)
    X_validation = codifica.transform(X_validation)
    X_train = codifica.transform(X_train)
    X_test = codifica.transform (X_test)
    gridmodel = GridSearchCV(estimator,params,scoring='f1_weighted')
    gridmodel.fit(X_trainval,y_trainval)
    print(gridmodel.best_params_)
    analysis(1,[name],[gridmodel.best_estimator_],codifica,ts=(X_train,y_train,X_test,y_test))


def Bayes_params(start,end,granularity):
    return {'alpha' : np.arange(start,end,granularity)}
def SMV_params(start,end,granularity):
    return {'C' : np.arange(start,end,granularity)}
def Logistic_params():
    return {'C' : [10,100,1000,10000,100000,1000000]}

if __name__ == "__main__":
    codif =  CountVectorizer(analyzer='char')
    namelist = ["Naive Bayes","SVM","Logistic regression"]
    classifiers = [Bayes_clasifier(),SVM_classifier(),LogisticRegression_classifier()]
    # GridModelSelection(0,codif,Bayes_clasifier(),Bayes_params(0.1,5,0.1),"Naive Bayes")
    #GridModelSelection(0,codif,LogisticRegression_classifier(),Logistic_params(),"Logistic Regression")
    #GridModelSelection(0,codif,SVM_classifier(),SMV_params(0.1,2,0.1),"Linear SVM")
    analysis(10,["Logistic regression"],[LogisticRegression_classifier(C_param=1000)],codif)