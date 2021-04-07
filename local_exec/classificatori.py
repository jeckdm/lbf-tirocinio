from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier,LogisticRegression
import sklearn
import numpy as np
import init
def naive_bayes(X_train,y_train,X_test,y_test):
    nb = Pipeline([('vect', CountVectorizer(analyzer='word')),
                ('tfidf', TfidfTransformer()),
                ('clf', MultinomialNB()),
                ])
    nb.fit(X_train, y_train)
    y_pred = nb.predict(X_test)
    print("naive bayes:")
    print('accuracy %s' % sklearn.metrics.accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))

def SVM (X_train,y_train,X_test,y_test):
    sgd = Pipeline([('vect', CountVectorizer(analyzer='word')),     #codica bag of words
                    ('tfidf', TfidfTransformer()),          # normalizzo la matrice, tecninca utilizzata nei documenti, non particolarmente
                                                            #significativa in questo caso probabilmente; rimuovendo questo passaggio non ho grossi
                                                            #cambiamenti a lv di performance
                    ('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42, max_iter=5, tol=None)), #creo classificatore ( parametri in base a quelli del sito di riferimento)
                ])
    sgd.fit(X_train, y_train)         #effetuo il training
    y_pred = sgd.predict(X_test)     #predico a partire da X_test
    print("SVM:")
    print('accuracy %s' % sklearn.metrics.accuracy_score(y_pred, y_test))  #calcolo accuratezza dal confronto fra y_pred e y_test
    print(classification_report(y_test, y_pred))                           # stampo report (accureatezza, precisione ,recall e f1 score)

def logistic_regression(X_train,y_train,X_test,y_test):
    logreg = Pipeline([('vect', CountVectorizer(analyzer='word')),
                ('tfidf', TfidfTransformer()),
                ('clf', LogisticRegression(n_jobs=1, C=1e5,max_iter=5000)),
               ])
    logreg.fit(X_train, y_train)
    y_pred = logreg.predict(X_test)
    print("logistic regression:")
    print('accuracy %s' % sklearn.metrics.accuracy_score(y_pred, y_test))
    print(classification_report(y_test, y_pred))

legitimate_URLs,phishing_URLs = init.load_data(1) #prendo i dati (argomento uno va ad indicare il ratio fra phishing e legitimate
                                                    #ho scelto di usare un dataset bilanciato)

X= legitimate_URLs+phishing_URLs
y= [0]*len(legitimate_URLs)+ [1]*len(phishing_URLs)      #associo ad ogni legit 0 e ad ogni phish 1
    
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.3)
if(len(X_train)!=len(set(X_train)-set(X_test))):  #verifico se sono presenti elementi uguali in test e train set
    raise AssertionError
naive_bayes(X_train,y_train,X_test,y_test)
SVM(X_train,y_train,X_test,y_test)         # qui sono commentati i vari passaggi, che poi si ripetono pressochè identici per ogni classificatore
logistic_regression(X_train,y_train,X_test,y_test)
