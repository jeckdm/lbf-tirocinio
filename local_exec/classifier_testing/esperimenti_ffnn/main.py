import sys
sys.path.append('.\local_exec')
import pandas as pd
import argparse
import init
from classifier_testing.esperimenti_ffnn import cross_validation as cv

def main(args):
    # Parametri
    ratio = args.ratiolp
    outer_folds = args.nfold[0]
    inner_folds = args.nfold[1]
    hidden_layer_sizes = args.neurons
    result_loc = args.resultloc
    learning_rates = args.learnrate

    # Carico dataset
    X, y = init.load_data()

    # Codifico (A priori non aggiungo informazioni)
    X_ff_encoded, y_ff_encoded, vocabulary = init.CV_encode(X, y)
    d = init.map_to_number(X)
    X_rnn_encoded, y_rnn_encoded = init.RNN_encode(X, y, d)

    # Model selection
    model_results, params_results = cv.model_selection(X_ff_encoded, y_ff_encoded, params = {'hidden_layer_size' : hidden_layer_sizes, 'learning_rate' : learning_rates}, outer_folds = outer_folds, inner_folds = inner_folds, ratio = ratio, njobs = -1)
    # Creo df risultati e salvo
    df_results = pd.DataFrame(model_results)
    df_resultsp = pd.DataFrame(params_results)

    with open(f"model_selection_{result_loc}.tex", "a") as file:
        file.write(df_results.round(3).to_latex(position = "H"))
        file.write(df_resultsp.to_latex(position = "H"))
    
    best_params = list(set(tuple(v.values()) for _,v in params_results.items()))

    # Risultati cv FF
    cv_mean_results = {str(p) : None for p in best_params}
    cv_std_results = {str(p) : None for p in best_params}

    for hidden_layer_size, learning_rate in best_params:
        cv_mean_results[str((hidden_layer_size, learning_rate))], cv_std_results[str((hidden_layer_size ,learning_rate))] = cv.nn_cross_validation( X_ff_encoded, y_ff_encoded, 
                                                                                                                                                    folds = outer_folds, 
                                                                                                                                                    ratio = ratio, 
                                                                                                                                                    hidden_layer_size = hidden_layer_size, 
                                                                                                                                                    learning_rate = learning_rate)

    with open(f"cross_validation_{result_loc}.tex", 'a') as file:
        file.write(pd.DataFrame(cv_mean_results).to_latex(position = "H"))
        file.write(pd.DataFrame(cv_std_results).to_latex(position = "H"))

    # Risultati cv RNN con parametri di default articolo
    cv_mean_results = {i : None for i in [16]}
    cv_std_results = {i : None for i in [16]}

    for hidden_layer_size in [16]:
        cv_mean_results[hidden_layer_size], cv_std_results[hidden_layer_size] = cv.rnn_cross_validation(X_rnn_encoded, y_rnn_encoded, outer_folds, ratio)

    with open(f"cross_validation_{result_loc}.tex", 'a') as file:
        file.write(pd.DataFrame(cv_mean_results).to_latex(position = "H"))
        file.write(pd.DataFrame(cv_std_results).to_latex(position = "H"))
    
    
if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--ratiolp", "-u", type = float, default = 0.2)
    parse.add_argument("--nfold", "-f", type = int, nargs = "+", default = [5,5])
    parse.add_argument("--neurons", "-n", type = int, nargs = "+")
    parse.add_argument("--learnrate", "-r", type = float, nargs = "+")
    parse.add_argument("--resultloc", "-l", type = str)

    args = parse.parse_args()

    main(args)