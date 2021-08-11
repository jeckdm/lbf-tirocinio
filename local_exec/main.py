import argparse
import pandas as pd
import torch
import numpy as np
import os
import matplotlib.pyplot as plt
import init, helpers, FFNN, analysis, helpers, RNN as R, trainRNN, BF
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

LBF_graph_params = {
        'title' : ['FNR Ratio', ' True FPR', 'Sizes'],
        'ylim' :  [(0.0, 1.0), (0.0, 0.039), (20, 110)],
        'ylabel' : ['Classifier False Negative Rate', 'Overall FPR', 'Total LBF Size (Kbyte)'],
        'legend' : ['medium', None, None]
    }

SLBF_graph_params = {
    'title' : ['FNR Ratio', ' True FPR', 'Sizes', 'Filter Sizes'],
    'ylim' :  [(0.0, 1.0), (0.0, 0.039), (20, 110), (-1, 80)],
    'ylabel' : ['Classifier False Negative Rate', 'Overall FPR', 'Total LBF Size (Kbyte)', 'Size (Kbyte)'],
    'legend' : [{'fontsize' : 'medium', 'ncol' : 1}, {'fontsize' : 'medium', 'ncol' : 1}, {'fontsize' : 'medium', 'ncol' : 1}, {'fontsize' : 'medium', 'ncol' : 2}]
}  

def create_train_FFNN(X_train, y_train, input_size, hidden_layer, learning_rate, save_loc):
    '''
    Crea e addestra una FFNN sul dataset in ingresso.
    Ritorna il modello addestrato e la sua dimensione
    '''

    es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 5)
    mc = ModelCheckpoint(f'{save_loc}/modelli/best_model.h5', monitor = 'val_loss', mode = 'min')
    LBF_X_train, LBF_y_train = helpers.shuffle(X_train, y_train)
    ffnn = FFNN.create_sequential(input_size = (input_size, ), hidden_layer_size = hidden_layer, learning_rate = learning_rate)
    FFNN.train(ffnn, LBF_X_train, LBF_y_train, epochs = 30, validation_split = 0.2, cbs = [es, mc])
    ffnn = load_model(f'{save_loc}/modelli/best_model.h5')
    model_size = FFNN.model_size(ffnn, f"{save_loc}/modelli/FFNN_({hidden_layer}, {learning_rate}).p", use_pickle = True)

    return ffnn, model_size

def tau_graph(probs0, probs1, BF_sizes, fprs, LBF_fpr_ratio, SLBF_fpr_ratio, prediction, legit_testing_list, model_size, phishing_list, name, save_loc):
    '''
    Testa il classificatore con diversi valori di tau sulla base degli fpr ed fpr ratio e la struttura in modo empirico
    Ritorna i risultati dei test empirici e genera i relatvi grafici
    '''

    # Analisi tau LBF
    false_negs, taus, fnrs = analysis.tau_analysis(probs0, probs1, fprs, LBF_fpr_ratio, phishing_list)
    fnrs_df, true_fpr_LBF, sizes_LBF, times_LBF, df_result_LBF = helpers.LBF_analysis(fprs, LBF_fpr_ratio, fnrs, false_negs, prediction, taus, legit_testing_list, model_size)
    helpers.generate_LBF_graphs(name, fnrs_df, true_fpr_LBF, BF_sizes, sizes_LBF / 1024, LBF_graph_params, save_loc = f"{save_loc}/grafici/{name}_LBF_risultati.png")

    # Analisi tau SLBF
    false_negs, taus, fnrs = analysis.tau_analysis(probs0, probs1, fprs, SLBF_fpr_ratio, phishing_list)
    fnrs_df, true_fpr_SLBF, BF_initial_sizes_SLBF, BF_backup_sizes_SLBF, sizes_SLBF, times_SLBF, df_result_SLBF = helpers.SLBF_analysis(fprs, SLBF_fpr_ratio, fnrs, false_negs, prediction, taus, legit_testing_list, phishing_list, model_size)
    helpers.generate_SLBF_graphs(name, fnrs_df, true_fpr_SLBF, BF_sizes , sizes_SLBF / 1024, BF_initial_sizes_SLBF / 1024, BF_backup_sizes_SLBF / 1024, SLBF_graph_params, save_loc = f"{save_loc}/grafici/{name}_SLBF_risultati.png")

    return true_fpr_LBF, true_fpr_SLBF, sizes_LBF, sizes_SLBF, df_result_LBF, df_result_SLBF

def comparison_graph(df_best_sizes, df_sizes, fprs, ffnn_par, rnn_par, save_loc):
    '''
    Genera il grafico di confronto delle dimensioni delle strutture al variare dell'fpr per ognuno dei classificatori testati.
    '''

    fig, axes = plt.subplots(1, 2, figsize = (13,5), sharex = True, sharey= True)

    df_best_sizes.loc['LBF'].xs('Totale', level = 1, drop_level = True).xs('FPR', axis = 1).T.plot(ax = axes[0] , marker = 'o', markeredgecolor = 'black', xticks = fprs, xlabel = 'False positive rate', ylabel = 'Sizes (Kbytes)', title = 'LBF Filters size comparison', zorder = 10)
    df_best_sizes.loc['SLBF'].xs('Totale', level = 1, drop_level = True).xs('FPR', axis = 1).T.plot(ax = axes[1] , marker = 'o', markeredgecolor = 'black', xticks = fprs, xlabel = 'False positive rate', ylabel = 'Sizes (Kbytes)', title = 'SLBF Filters size comparison', zorder = 10)

    for par in ffnn_par:
        axes[0].scatter(df_sizes['LBF'][f'FFNN{par}']['x'], df_sizes['LBF'][f'FFNN{par}']['y'], marker = '.')
        axes[1].scatter(df_sizes['SLBF'][f'FFNN{par}']['x'], df_sizes['SLBF'][f'FFNN{par}']['y'], marker = '.')

    for par in rnn_par:
        axes[0].scatter(df_sizes['LBF'][f'RNN{par}']['x'], df_sizes['LBF'][f'RNN{par}']['y'], marker = '.')
        axes[1].scatter(df_sizes['SLBF'][f'RNN{par}']['x'], df_sizes['SLBF'][f'RNN{par}']['y'], marker = '.')

    plt.savefig(f"{save_loc}/grafici/comparison_graph.png")

def update_dfs(df, df_result_LBF, df_result_SLBF, scatter_dict, true_fpr_SLBF, sizes_SLBF, true_fpr_LBF, sizes_LBF, par, fpr, name):
    scatter_dict['SLBF'][name]['x'].append(true_fpr_SLBF[fpr].tolist())
    scatter_dict['SLBF'][name]['y'].append((sizes_SLBF[fpr] / 1024).tolist())
    scatter_dict['LBF'][name]['x'].append(true_fpr_LBF[fpr].tolist())
    scatter_dict['LBF'][name]['y'].append((sizes_LBF[fpr] / 1024).tolist())

    df.loc[('LBF', name, 'Totale'), ('FPR', fpr)] = df_result_LBF[fpr]['size']
    df.loc[('LBF', name, 'Classificatore'), ('FPR', fpr)] = df_result_LBF[fpr]['model-size']
    df.loc[('SLBF', name, 'Totale'), ('FPR', fpr)] = df_result_SLBF[fpr]['size']
    df.loc[('SLBF', name, 'Classificatore'), ('FPR', fpr)] = df_result_SLBF[fpr]['model-size']
    df.loc[('SLBF', name, 'Backup'), ('FPR', fpr)] = df_result_SLBF[fpr]['backup-size']
    df.loc[('SLBF', name, 'Iniziale'), ('FPR', fpr)] = df_result_SLBF[fpr]['initial-size']

    return df, scatter_dict

def main(args): 
    FFNN_params = [(hidden, lr) for hidden in args.ffhidden for lr in args.fflearning]
    FFNN_bin_params = [(hidden, lr) for hidden in args.ffbinhidden for lr in args.fflearning]
    RNN_params = args.rnnpar
    fprs = args.fprs
    LBF_fpr_ratio = args.lbfratio
    SLBF_fpr_ratio = args.slbfratio
    save_loc = args.resultloc
    data_loc = args.dataloc

    if not os.path.isdir(save_loc): 
        os.makedirs(f"{save_loc}/grafici")
        os.makedirs(f"{save_loc}/modelli")
        os.makedirs(f"{save_loc}/tabelle")

    # Carico dataset
    X, y = init.load_data(data_loc)

    # Codifica CV
    X_ff_enc, _, v = init.CV_encode(X, y)
    X_ff_bin_enc, _ = init.bin_encode(X, y, init.map_to_number(X))
    X_rnn_enc, _ = init.RNN_encode(X, y, init.map_to_number(X))
    device = 'cpu'

    # Splitting dataset
    LBF_X_train, LBF_y_train, LBF_X_test, LBF_y_test, phishing_list, legit_testing_list = init.LBF_train_test_split(X, y, X_ff_enc)

    BF_sizes = {fpr : BF.BloomFilter.get_size(len(phishing_list) / 8192, fpr) for fpr in fprs}

    names = ['FFNN'] * len(FFNN_params) + ['RNN'] *  len(RNN_params) + ['FFNN_Bin'] * len(FFNN_params)
    names = [f'{names[i]}{par}' for i,par in enumerate(FFNN_params + RNN_params + FFNN_bin_params)]

    # Creazione Dataframe per risultati
    idx =  [('BF', '', 'Totale')] + [('LBF', n, 'Totale') for n in names] + [('LBF', n, 'Classificatore') for n in names] + [ ('SLBF', n, 'Totale') for n in names] + [('SLBF', n, 'Classificatore') for n in names] + [('SLBF', n, 'Iniziale') for n in names] + [('SLBF', n, 'Backup') for n in names]
    columns = [('FPR', fpr) for fpr in fprs]
    dataframe_result = pd.DataFrame(
        np.random.randn(1 + 2 * len(FFNN_params + RNN_params + FFNN_bin_params) + 4 * len(FFNN_params + RNN_params + FFNN_bin_params), len(fprs)),
        index = pd.MultiIndex.from_tuples(sorted(idx)),
        columns = pd.MultiIndex.from_tuples(columns)
    )
    dataframe_result.index.names = ['Struttura', 'Classificatore', 'Dimensioni']

    scatter_result = {
        'LBF': {par : {'x' : [], 'y' : []} for par in names}, 'SLBF' : {par : {'x' : [], 'y' : []} for par in names}
    }

    for fpr in fprs:
        dataframe_result.loc[('BF', '', 'Totale'), ('FPR', fpr)] = BF_sizes[fpr]
    
    for par in FFNN_params:
        # Creazione modello FFNN
        ffnn, model_size = create_train_FFNN(LBF_X_train, LBF_y_train, len(v), par[0], par[1], save_loc)
        # Predizioni su test FFNN
        probs0, probs1 = FFNN.get_classifier_probs(ffnn, LBF_X_train, LBF_y_train)
        prediction = ffnn.predict(LBF_X_test) 

        # Analisi tau e generazione grafici
        true_fpr_LBF, true_fpr_SLBF, sizes_LBF, sizes_SLBF, df_result_LBF, df_result_SLBF = tau_graph(probs0, probs1, BF_sizes, fprs, LBF_fpr_ratio, SLBF_fpr_ratio, prediction, legit_testing_list, model_size, phishing_list, f"FFNN_{par}", save_loc)

        # Salvo risultati migliori
        for fpr in fprs:
            dataframe_result, scatter_result = update_dfs(dataframe_result, df_result_LBF, df_result_SLBF, scatter_result, true_fpr_SLBF, sizes_SLBF, true_fpr_LBF, sizes_LBF, par, fpr, f"FFNN{par}")
    
    LBF_X_train, LBF_y_train, LBF_X_test, LBF_y_test, phishing_list, legit_testing_list = init.LBF_train_test_split(X, y, X_ff_bin_enc)
    for par in FFNN_bin_params:
        # Creazione modello FFNN
        ffnn, model_size = create_train_FFNN(LBF_X_train, LBF_y_train, 210, par[0], par[1], save_loc)
        # Predizioni su test FFNN
        probs0, probs1 = FFNN.get_classifier_probs(ffnn, LBF_X_train, LBF_y_train)
        prediction = ffnn.predict(LBF_X_test) 

        # Analisi tau e generazione grafici
        true_fpr_LBF, true_fpr_SLBF, sizes_LBF, sizes_SLBF, df_result_LBF, df_result_SLBF = tau_graph(probs0, probs1, BF_sizes, fprs, LBF_fpr_ratio, SLBF_fpr_ratio, prediction, legit_testing_list, model_size, phishing_list, f"FFNN_Bin_{par}", save_loc)

        # Salvo risultati migliori
        for fpr in fprs:
            dataframe_result, scatter_result = update_dfs(dataframe_result, df_result_LBF, df_result_SLBF, scatter_result, true_fpr_SLBF, sizes_SLBF, true_fpr_LBF, sizes_LBF, par, fpr, f"FFNN_Bin{par}")

    LBF_X_train, LBF_y_train, LBF_X_test, LBF_y_test, phishing_list, legit_testing_list = init.LBF_train_test_split(X, y, X_rnn_enc)

    for par in RNN_params:
        # Creazione modello RNN
        rnn = R.RNN(emb_size = 5, h_size = par, layers = 1).to(device) 
        trainRNN.train(rnn, torch.tensor(LBF_X_train), torch.tensor(LBF_y_train), optimizer = torch.optim.Adamax(rnn.parameters()), device = device)
        model_size = R.model_size(rnn, f"{save_loc}/modelli/RNN_{par}.p", use_pickle = True)
        rnn = R.load_pickle_model(f"{save_loc}/modelli/RNN_{par}.p", h_size = par)

        # Predizioni su test RNN
        probs1, probs0 = trainRNN.get_classifier_probs(rnn, torch.tensor(LBF_X_train), torch.tensor(LBF_y_train))
        y_hat, _ = rnn(torch.tensor(LBF_X_test).to(device))
        prediction = torch.sigmoid(y_hat[:,:,1])[:,149].squeeze().detach().cpu().numpy()

        # Analisi tau e generazione grafici
        true_fpr_LBF, true_fpr_SLBF, sizes_LBF, sizes_SLBF, df_result_LBF, df_result_SLBF = tau_graph(probs0, probs1, BF_sizes, fprs, LBF_fpr_ratio, SLBF_fpr_ratio, prediction, legit_testing_list, model_size, phishing_list, f"RNN_{par}", save_loc)

        # Salvo risultati migliori
        for fpr in fprs:
            dataframe_result, scatter_result = update_dfs(dataframe_result, df_result_LBF, df_result_SLBF, scatter_result, true_fpr_SLBF, sizes_SLBF, true_fpr_LBF, sizes_LBF, par, fpr, f"RNN{par}")

    # Genero grafico di confronto spazio tra RNN e FFNN
    comparison_graph(dataframe_result, scatter_result, fprs, FFNN_params, RNN_params, save_loc)
    
    # Salvo tabella risultati
    with open(f'{save_loc}/tabelle/results.tex', 'a') as file:
        file.write(dataframe_result.round(3).to_latex(
            column_format = '|l|l|l|c|c|c|c|', 
            multicolumn_format = '|c|',  
            multirow = True, 
            float_format="%.2f", 
            bold_rows = True, 
            position = 'H').replace('\\toprule', '\\hline').replace('\\midrule', '\\hline').replace('\\bottomrule','\\hline'))

if __name__ == "__main__":
    parse = argparse.ArgumentParser()

    parse.add_argument("resultloc", type = str)
    parse.add_argument("dataloc", type = str)
    parse.add_argument("fprs", type = float, nargs = "?", default = [0.001, 0.005, 0.01, 0.02])
    parse.add_argument("ffhidden", type = int, nargs = "?", default = [30, 20])
    parse.add_argument("ffbinhidden", type = int, nargs = "?", default = [10, 8])
    parse.add_argument("fflearning", type = int, nargs = "?", default = [0.001])
    parse.add_argument("rnnpar", type = int, nargs = "?", default = [16, 8, 4])
    parse.add_argument("lbfratio", type = float, nargs = "?", default = [.1 * i for i in range(1, 11)])
    parse.add_argument("slbfratio" , type = float, nargs = "?", default = [1. * i for i in range(1, 11)])

    args = parse.parse_args()

    main(args)