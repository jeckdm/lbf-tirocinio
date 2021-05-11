from sklearn.naive_bayes import MultinomialNB
import sys
sys.path.append('local_exec/')
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
from classificatori import helpers,train_and_val,model_selection
nfeatureword = 750
nfeaturechar = 0.90

if __name__ == "__main__" :    
    paramcodif= "word"
    codif =  CountVectorizer(analyzer=paramcodif,dtype=float)
    namelist,classifier_list = helpers.get_default_classifier_list(SVM=True)
    
    #train_and_val.Cross_Validation_analisys(classifier_list,namelist,codif,componenti=nfeatureword)
    #model_selection.ModelSelection(codif, Bayes_clasifier(), select_params(-10,10,"alpha"), "Naive bayes","alpha",multilevel=True)
    params = model_selection.randomize_params()
    model_selection.ModelSelection(codif,classifier_list.pop(),params,namelist.pop(),["C","gamma"],Randomize=True)
    #model_selection.ModelSelection(codif,classifier_list.pop(),params,namelist.pop(),["gamma"],multilevel=True)    
    #analysis(10, SVMnamelist,svm_try, codif) 
    #analysis(10, namelist, classifiers, codif,pca=True,small=False)
    
