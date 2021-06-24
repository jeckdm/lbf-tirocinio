# Syspath
import sys

from tensorflow.keras import callbacks

sys.path.append('.\local_exec')

import init
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from classifier_testing.esperimenti_ffnn import FFNN as ff
from classifier_testing.esperimenti_ffnn import helpers
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint

def main(): 
    # Carico dataset
    X, y = init.load_data()
    
    # Codifica con CountVect
    X_ff_enc, y_ff_enc, vocabulary = init.CV_encode(X, y)

    LBF_X_train, LBF_y_train, LBF_X_test, LBF_y_test, phishing_list, legit_testing_list = init.LBF_train_test_split(X, y, X_ff_enc)
    LBF_X_train, LBF_y_train = helpers.shuffle(X_ff_enc, y_ff_enc)
    es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 5)
    mc = ModelCheckpoint('local_exec/classifier_testing/esperimenti_ffnn/risultati/best_model.h5', monitor = 'val_loss', mode = 'min')

    params = {'hidden_layer_size' : [10, 20, 25, 30, 35], 'learning_rate' : [0.0001, 0.001, 0.01]}
    model = KerasClassifier(build_fn = ff.create_sequential, input_size = (82, ), epochs = 30, batch_size = 128, verbose = 0)
    grid = GridSearchCV(model, params, scoring = 'f1', cv = 5, n_jobs = 8).fit(LBF_X_train, LBF_y_train)

    best_model = grid.best_estimator_
    best_params = grid.best_params_

    print("Best: %f using %s" % (grid.best_score_, grid.best_params_))
    means = grid.cv_results_['mean_test_score']
    stds = grid.cv_results_['std_test_score']
    params = grid.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

if __name__ == "__main__":
    main()