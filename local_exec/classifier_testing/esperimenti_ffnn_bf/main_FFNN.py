# Syspath
import sys

from tensorflow.keras import callbacks

sys.path.append('.\local_exec')

# Librerie
import init
import trainRNN
import RNN as R
import pandas as pd
import argparse
import torch
from classifier_testing.esperimenti_ffnn import FFNN
from classifier_testing.esperimenti_ffnn import trainRNN
from classifier_testing.esperimenti_ffnn_bf import analysis
from classifier_testing.esperimenti_ffnn_bf import helpers
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
import numpy as np

def shuffle(X, y, seed = 0):
    """ Ritorna una permutazione di X,y generata con il seed in input"""

    np.random.seed(seed)
    perm = np.random.permutation(len(X))

    X = X[perm]
    y = y[perm]

    return X, y

def main(args): 
    # Parametri FFNN
    params = (30, 0.001)
    # Base sizes
    BF_sizes = {0.001 : 78616.625, 0.005 : 60299.75, 0.01 : 52411.0, 0.02 : 44522.375}
    # FPR test
    fprs = [0.001,0.005,0.01,0.02]
    fpr_ratios = [0.1*i for i in range(1,11)] # Ratio per LBF
    fpr_ratios2 = [1.*i for i in range(1,11)] # Ratio per SLBF

    # Carico dataset
    X, y = init.load_data()
    
    # Codifica con CountVect
    X_ff_enc, y_ff_enc, vocabulary = init.CV_encode(X, y)

    LBF_X_train, LBF_y_train, LBF_X_test, LBF_y_test, phishing_list, legit_testing_list = init.LBF_train_test_split(X, y, X_ff_enc)

    # Creazione modelli
    es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 5)
    mc = ModelCheckpoint('local_exec/classifier_testing/risultati/models/best_model.h5', monitor = 'val_loss', mode = 'min')
    LBF_X_train, LBF_y_train = shuffle(LBF_X_train, LBF_y_train)
    ffnn = FFNN.create_sequential(input_size = (len(vocabulary), ), hidden_layer_size = params[0], learning_rate = params[1])
    FFNN.train(ffnn, LBF_X_train, LBF_y_train, epochs = 30, validation_split = 0.2, cbs = [es, mc])
    ffnn = load_model('local_exec/classifier_testing/risultati/models/best_model.h5')
    model_size = FFNN.model_size(ffnn, f"local_exec/classifier_testing/risultati/models/FFNN_{params}.p")

    # Predizioni su test FFNN
    probs0, probs1 = FFNN.get_classifier_probs(ffnn, LBF_X_train, LBF_y_train)
    prediction = ffnn.predict(LBF_X_test)

    # Analisi tau LBF
    false_negs, taus, fnrs = analysis.tau_analysis(probs0, probs1, fprs, fpr_ratios, phishing_list)
    # Trasformo fnrs in un df (Da inserire in qualche altra func)
    fnrs_df = pd.DataFrame(index=fpr_ratios, columns=fprs)
    for fpr in fprs:
      for fpr_ratio in fpr_ratios:
        fnrs_df.loc[fpr_ratio,fpr] = fnrs[(fpr, fpr_ratio)]

    helpers.create_graph(fnrs_df,f"FFNN {params}", path = f"local_exec/classifier_testing/risultati/plots/fnrs_LBF_{params}.png", xlabel = "Classifier FPR ratio "+r"$\epsilon_\tau/\epsilon$", ylabel = "Classifier FNR")

    true_fpr_LBFs, sizes_LBFs = helpers.LBF_analysis(fprs, fpr_ratios, false_negs, prediction, taus, legit_testing_list, model_size, BF_sizes, graph_name = f"FFNN_{params}")

    # Analisi tau SLBF
    false_negs, taus, fnrs = analysis.tau_analysis(probs0, probs1, fprs, fpr_ratios + fpr_ratios2, phishing_list)
    # Trasformo fnrs in un df (Da inserire in qualche altra func)
    fnrs_df = pd.DataFrame(index=fpr_ratios + fpr_ratios2, columns=fprs)
    for fpr in fprs:
      for fpr_ratio in fpr_ratios + fpr_ratios2:
        fnrs_df.loc[fpr_ratio,fpr] = fnrs[(fpr, fpr_ratio)]

    helpers.create_graph(fnrs_df,f"FFNN {params}", path = f"local_exec/classifier_testing/risultati/plots/fnrs_SLBF_{params}.png", xlabel = "Classifier FPR ratio "+r"$\epsilon_\tau/\epsilon$", ylabel = "Classifier FNR")

    true_fpr_SLBFs, sizes_SLBFs = helpers.SLBF_analysis(fprs, fpr_ratios + fpr_ratios2, false_negs, prediction, taus, legit_testing_list, phishing_list, model_size, BF_sizes, graph_name = f"FFNN_{params}")

    helpers.create_comparison_size_graph(true_fpr_LBFs, true_fpr_SLBFs, sizes_LBFs, sizes_SLBFs, path = f"local_exec/classifier_testing/risultati/plots/filters_comparison_{params}.png")

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--fprs", "-f", type = float, nargs = "+")
    parse.add_argument("--ratios", "-r", type = float, nargs = "+")
    parse.add_argument("--resultloc", "-l", type = str)

    args = parse.parse_args()

    main(args)