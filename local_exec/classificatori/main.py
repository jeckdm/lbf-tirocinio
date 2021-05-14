import sys
sys.path.append('local_exec/')
from sklearn.feature_extraction.text import CountVectorizer
from classificatori import helpers,train_and_val,model_selection,size
nfeatureword = 750
nfeaturechar = 0.90

if __name__ == "__main__" :    
    paramcodif= "char"
    codif =  CountVectorizer(analyzer=paramcodif,dtype=float)
    namelist,classifier_list = helpers.get_default_classifier_list(SVM=True)
    train_and_val.Cross_Validation_analisys(classifier_list,namelist,codif,pre_codif= False,save=True)
    #size.try_Naive_bayes()
    #model_selection.ModelSelection(codif, classifier_list.pop(), model_selection.grid_params(-1,6,"C"), "Logistic Regression",["C"],multilevel=False)
    #params = model_selection.randomize_params()
    #model_selection.ModelSelection(codif,classifier_list.pop(),params,namelist.pop(),["C","gamma"],Randomize=True)
    #model_selection.ModelSelection(codif,classifier_list.pop(),params,namelist.pop(),["gamma"],multilevel=True)    
    #analysis(10, SVMnamelist,svm_try, codif) 
    #analysis(10, namelist, classifiers, codif,pca=True,small=False)score=0.973
    