from operator import mod
import pickle
import sys
import os
import joblib
from sklearn.feature_extraction.text import  TfidfTransformer
from sklearn.model_selection import train_test_split , StratifiedKFold
import numpy as np
sys.path.append('local_exec/')
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer
from classificatori import helpers,do_codif
latec_path = "/local_exec/classificatori/saved_mode/size_analysis.tex"

def Model_saved(model,attributes,path,Numpy=True,Joblib=True,Pickle=True):
    if(Numpy):
        print(attributes)
        np.save(path+"numpy",attributes) 
    if Pickle:
        with open(path+"pickle.pk1",'wb') as file:
            pickle.dump(model,file)
    if Joblib:
        with open(path+"joblib.pk1",'wb') as file:
            joblib.dump(model,file)

def save_Naive_Bayes(model,path):
    log_prob = model.feature_log_prob_
    class_log = model.class_log_prior_
    np.save(path+"log_prob",np.array(log_prob,dtype=np.float16))
    np.save(path+"class_log",np.array(log_prob,dtype=np.float16))
    size = os.path.getsize(path+"log_prob" + ".npy")
    size = size + os.path.getsize(path+"class_log"+ ".npy")    
    return size

def save_Linear_Logistic(model,path):
    coef = model.coef_
    intercept = model.intercept_
    np.save(path+"coef",np.array(coef,dtype=np.float16))
    np.save(path+"intercept",np.array(intercept,dtype=np.float16))
    size = os.path.getsize(path+"coef"+".npy")
    size = size + os.path.getsize(path+"intercept" + ".npy")
    return size

def save_tdf(tdf,path):
    vect = tdf.idf_
    path = path + "_idf"
    np.save(path, np.array(vect,dtype=np.float16))
    size = os.path.getsize(path+".npy")
    return size

def save_SVM(model,path):
    support_vector = model.support_vectors_
    intercept = model.intercept_
    class_weight = model.class_weight_
    dual_coef = model.dual_coef_
    shape_fit = model.shape_fit_
    sparse = model._sparse
    probA = model.probA_
    probB = model.probB_
    n_support = model.n_support_
    gamma = model._gamma
    np.save(path+"_support_vector", np.array(support_vector,dtype=np.float16))
    np.save(path+"_intercept_", np.array(intercept,dtype=np.float16))
    np.save(path+"_class_weight", np.array(class_weight,dtype=np.float16))
    np.save(path+ "_dual_coef", np.array(dual_coef,dtype=np.float16))
    np.save(path+ "_shape_fit", np.array(shape_fit,dtype=np.float16))
    np.save(path+ "_sparse", np.array(sparse,dtype=np.float16))
    np.save(path+ "_probA", np.array(probA,dtype=np.float16))
    np.save(path+ "_probB", np.array(probB,dtype=np.float16))
    np.save(path+ "_n_support", np.array(n_support,dtype=np.float16))
    np.save(path+ "gamma", np.array(gamma,dtype=np.float16))
    
def save_Truncated_SVD (model,path):
    componenti = model.components_
    path = path+"_SVD_size"
    np.save(path,componenti)
    size = os.path.getsize(path+".npy")
    print(size)
    return size


def load_and_val_SVM(X_train,X_test,y_test,path):
    coef = np.load(path+"coef.npy")
    print("dopo: ",coef[:10])
    intercept = np.load(path+"intercept.npy")
    print("dopo : ", intercept)
    tdf = TfidfTransformer()
    tdf.fit(X_train)
    X_test = tdf.transform(X_test)
    new_model = helpers.SVM_linear_classifier()
    new_model.intercept_ = intercept
    new_model.coef_ = coef
    new_model.classes_ = np.array([0,1])
    y_pred = new_model.predict(X_test)
    print(classification_report(y_pred,y_test))

def try_logistic_regression():
    paramcodif= "word"
    codif =  CountVectorizer(analyzer=paramcodif,dtype=float)
    X_train,y_train,X_test,y_test=helpers.get_set_holdout()
    X_train,X_test,_,_= do_codif.codificate(X_train,X_test,codif,"")
    model = helpers.LogisticRegression_classifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_pred,y_test))
    coef = np.array(model.coef_,dtype=np.float16)
    intercept = np.array(model.intercept_,dtype=np.float16)
    print(intercept)
    new_model = helpers.LogisticRegression_classifier()
    new_model.intercept_ = intercept
    new_model.coef_ = coef
    new_model.classes_ = np.array([0,1])
    y_pred = new_model.predict(X_test)
    print(classification_report(y_pred,y_test))


def try_Linear_SVM():
    paramcodif= "word"
    codif =  CountVectorizer(analyzer=paramcodif,dtype=float)
    X_train,y_train,X_test,y_test=helpers.get_set_holdout()
    X_train,X_test,_= do_codif.codificate(X_train,X_test,codif,"")
    model = helpers.SVM_linear_classifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    coef = model.coef_
    intercept = model.intercept_
    new_model = helpers.SVM_linear_classifier()
    new_model.intercept_ = intercept
    new_model.coef_ = coef
    new_model.classes_ = np.array([0,1])
    y_pred = new_model.predict(X_test)
    print(classification_report(y_pred,y_test))



def try_SVM():
    paramcodif= "char"
    codif =  CountVectorizer(analyzer=paramcodif,dtype=float)
    X_train,y_train,X_test,y_test=helpers.get_set_holdout()
    X,_,y,_ = train_test_split(X_train,y_train,train_size=0.1)
    print(len(X))
    X_train,X_test,y_train,y_test = train_test_split(X,y)
    X_train,X_test,_= do_codif.codificate(X_train,X_test,codif,"")
    model = helpers.SVM_classifier()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    print(classification_report(y_pred,y_test))
    support_vector = model.support_vectors_
    intercept = model.intercept_
    class_weight = model.class_weight_
    dual_coef = model.dual_coef_
    shape_fit = model.shape_fit_
    sparse = model._sparse
    probA = model.probA_
    probB = model.probB_
    n_support = model.n_support_
    gamma = model._gamma

    print(intercept)
    new_model = helpers.SVM_classifier()
    new_model._intercept_ = intercept
    new_model.support_vectors_ = support_vector
    new_model.classes_ = np.array([1,0])
    new_model._dual_coef_ = dual_coef
    #new_model.fit_status_= 0
    new_model.class_weight_ = class_weight
    new_model._probA = probA
    new_model._probB = model.probB_
    new_model._n_support = model.n_support_
    new_model.shape_fit_ = shape_fit
    new_model._sparse = sparse
    new_model._gamma = model._gamma
    y_pred = new_model.predict(X_test)
    print(classification_report(y_pred,y_test))

