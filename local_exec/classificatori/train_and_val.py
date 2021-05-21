from numpy.core.arrayprint import ComplexFloatingFormat
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import sklearn
import sys
sys.path.append('local_exec/')
import numpy as np
from classificatori import helpers,do_codif,size
savepath = "/home/dav/Scrivania/latex/analysis"
modelPath = "local_exec/classificatori/saved_model"
metrics = ['precision', 'recall', 'f1-score', 'accuracy']

def train_and_fit(X_train, y_train, X_test, y_test, classifier, name, verbose=True,save = False):  #classe per fare training e valutare il classificatore  -> train and set
    nb = classifier
    nb.fit(X_train,  y_train)                                    #tfidf su X_train e training del classificatore
    y_pred = nb.predict(X_test)                                 #tfidf su X_test e calcollo di y pred
    if(verbose):
        print(f"{name}:")
        print('accuracy %s' % sklearn.metrics.accuracy_score(y_test, y_pred))
        print(classification_report(y_test,  y_pred))

    return classification_report(y_test, y_pred, output_dict=True,target_names=["Legitimate","Phishing"]),nb    # return classification report
'''
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

    helpers.makedf(rlist, metrics, classifier_name,(iteration>1))
'''

def Cross_Validation_analisys(classifier_list,classifier_name,codif,componenti=None,frel=False,verbose=False, pre_codif = False, save = False):
    X,y,kf = helpers.get_set_stratified()
   
    svname = codif.analyzer
    if(pre_codif):
        codif.fit(X)
        svname = svname+'_pre_codif'
    idf_size = [] 
    lsa_size = []
    rlist = {}
    dictsize = {}
    for name in classifier_name:
        rlist[name]= {}
        if(size):
            dictsize[name] = []
        for metric in metrics:
            rlist[name][metric] = []
    
    for train,test in kf.split(X,y):    
        X_test, X_train = X[test], X[train]
        y_test, y_train = y[test], y[train]
        assert(len(X_test)==len(list(set(X_test)-set(X_train))))
        X_train, X_test, svname, (tdf,lsa)= do_codif.codificate(X_train, X_test, codif, svname,frel,componenti,pre_codif=pre_codif)
        for classifier, name in zip(classifier_list, classifier_name):
            res,model = train_and_fit(X_train, y_train, X_test, y_test, classifier, name, verbose)
            for metric in metrics:
                if(metric == 'accuracy'):
                    rlist[name][metric].append((res[metric]))
                else:  
                    rlist[name][metric].append(res['Phishing'][metric])
            if(save):
                if(name == "Bayes"):
                    dictsize[name].append(size.save_Naive_Bayes(model,modelPath + "/naive_Bayes/" + svname ))
                if(name == "Linear SVM"):
                    dictsize[name].append(size.save_Linear_Logistic(model,modelPath + "/linear_SVM/" + svname ))
                if(name == "Logistic Regression"):
                    dictsize[name].append(size.save_Linear_Logistic(model,modelPath + "/logistic_regression/" + svname ))
                if(name == "SVM"):
                    dictsize[name].append(size.save_SVM(model,modelPath + "/SVM/" + svname ))
        if(save):
            idf_size.append(size.save_tdf(tdf,modelPath + "/"+ svname))
            if(componenti):
                lsa_size.append(size.save_Truncated_SVD(lsa,modelPath + "/" + svname))


    if(save):
        helpers.makedf(dictsize,classifier_name, modelPath + "/" + svname + svname + "size_report ")
        print(f"tdf ---> media : {np.average(idf_size)} dev_std : {np.std(idf_size)}")
    helpers.makedf(rlist, metrics, savepath + "_CRV" + svname, classifier_name)
