import sys
sys.path.append('.\local_exec')

import pandas as pd
import argparse
import init
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from classifier_testing.esperimenti_ffnn import cross_validation as cv

def save_df(dicts, result_name, approx = 3):
    """ Salva come tab latex i dizionari presenti in dicts """
    with open(result_name, 'a') as file:
        for d in dicts:
            print(d)
            file.write(pd.DataFrame(d).round(approx).to_latex(position = "H"))


def save_model_selection(X, y, params, outer_folds = 5, inner_folds = 5, ratio = 0.2, n_jobs = -1, epochs = 30, result_name = "risultati"):
    """ GridSearch sul modello ffnn con griglia = params, ritorna lista contenente i migliori parametri risultanti """
    # Model selection
    model_results, params_results = cv.model_selection(X, y, params = params, outer_folds = outer_folds, inner_folds = inner_folds, epochs = epochs, ratio = ratio, njobs = n_jobs)
    # Migliori parametri
    best_params = list(set(tuple(v.values()) for _,v in params_results.items()))

    if result_name is not None: 
        # Creo df risultati e salvo
        save_df([model_results, params_results], f"local_exec/classifier_testing/esperimenti_ffnn/risultati/model_selection_{result_name}.tex")

    return best_params

def save_cross_vals_nn(X, y, params, folds = 5, ratio = 0.2, callbacks = None, epochs = 30, validation_split = 0.2, result_name = "risultati"):
    """ Salva risultati cross validation con modelli aventi iperparametri (Neuroni, Learning rate) contenuti in params """
    # Risultati cv FF
    cv_mean_results = {str(p) : None for p in params}
    cv_std_results = {str(p) : None for p in params}

    # Da cambiare se aggiungo più parametri su cui faccio model selection
    for hidden_layer_size, learning_rate in params:
        cv_mean_results[str((hidden_layer_size, learning_rate))], cv_std_results[str((hidden_layer_size ,learning_rate))] = cv.nn_cross_validation(
            X, y, 
            folds = folds, 
            ratio = ratio, 
            hidden_layer_size = hidden_layer_size, 
            learning_rate = learning_rate, 
            callbacks = callbacks, 
            validation_split = validation_split,
            epochs = epochs
        )
    
    if result_name is not None:
        save_df([cv_mean_results, cv_std_results], f"local_exec/classifier_testing/esperimenti_ffnn/risultati/cross_validation_{result_name}.tex")


def main(args):
    # Parametri
    hidden_layer_sizes = args.neurons
    result_name = args.resultloc
    learning_rates = args.learnrate

    # Carico dataset
    X, y = init.load_data()

    # Codifico (A priori non aggiungo informazioni)
    X_ff_encoded, y_ff_encoded, vocabulary = init.CV_encode(X, y)
    d = init.map_to_number(X)
    X_rnn_encoded, y_rnn_encoded = init.RNN_encode(X, y, d)

    # Funzioni callback keras
    es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 5)
    mc = ModelCheckpoint('local_exec/classifier_testing/esperimenti_ffnn/risultati/best_model.h5', monitor = 'val_loss', mode = 'min')

    # Grid
    params = {'hidden_layer_size' : hidden_layer_sizes, 'learning_rate' : learning_rates}

    # Model selection
    best_params = save_model_selection(X_ff_encoded, y_ff_encoded, params, result_name = result_name, epochs = 1)
    
    # FF cross validation sui migliori parametri ottenuti
    save_cross_vals_nn(X_ff_encoded, y_ff_encoded, best_params, callbacks = [es, mc], result_name = result_name, epochs= 10)

    # Risultati cv RNN con parametri di default articolo
    cv_mean_results = {'16' : None}
    cv_std_results = {'16' : None}

    cv_mean_results['16'], cv_std_results['16'] = cv.rnn_cross_validation(X_rnn_encoded, y_rnn_encoded)

    save_df([cv_mean_results, cv_std_results], f"local_exec/classifier_testing/esperimenti_ffnn/risultati/cross_validation_{result_name}.tex")
    
if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--neurons", "-n", type = int, nargs = "+")
    parse.add_argument("--learnrate", "-l", type = float, nargs = "+")
    parse.add_argument("--resultloc", "-r", type = str)

    args = parse.parse_args()

    main(args)