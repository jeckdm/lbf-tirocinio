import argparse
import analysis
import init 
import BF
import torch
import numpy as np
import pandas as pd
import time
import pickle
import init, helpers, FFNN, analysis, helpers, RNN as R, trainRNN, BF
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from keras.models import load_model

def create_train_FFNN(X_train, y_train, input_size, hidden_layer, learning_rate, save_loc):
    es = EarlyStopping(monitor = 'val_loss', mode = 'min', patience = 5)
    mc = ModelCheckpoint(f'{save_loc}/modelli/best_model_{hidden_layer}.h5', monitor = 'val_loss', mode = 'min')
    LBF_X_train, LBF_y_train = helpers.shuffle(X_train, y_train)
    ffnn = FFNN.create_sequential(input_size = (input_size, ), hidden_layer_size = hidden_layer, learning_rate = learning_rate)
    FFNN.train(ffnn, LBF_X_train, LBF_y_train, epochs = 30, validation_split = 0.2, cbs = [es, mc])
    ffnn = load_model(f'{save_loc}/modelli/best_model_{hidden_layer}.h5')
    model_size = FFNN.model_size(ffnn, f"{save_loc}/modelli/FFNN_({hidden_layer}, {learning_rate}).p", use_pickle = True)

    return ffnn, model_size

def main(args): 
    fprs = args.fprs
    LBF_ratios = args.lbfratio
    SLBF_ratios = args.slbfratio

    X, y = init.load_data("data");
    X_ff_enc, _, v = init.CV_encode(X, y)
    # X_ff_bin_enc, _ = init.bin_encode(X, y, init.map_to_number(X))
    # X_rnn_enc, _ = init.RNN_encode(X, y, init.map_to_number(X))
    LBF_X_train, LBF_y_train, LBF_X_test, LBF_y_test, phishing_list, legit_testing_list = init.LBF_train_test_split(X, y, X_ff_enc)

    device = 'cpu'

    '''
    for fpr in fprs:
        tempi = []
        for _ in range(5):
            _, _, tempo = BF.run_BF(fpr, phishing_list, legit_testing_list)
            print(f"Tempo: {tempo}")
            tempi.append(tempo)
        with open(f'risultati/BloomFilter.tex', 'a') as file:
            file.write(f"Tempo medio: {np.mean(tempi)}, Dev st. : {np.std(tempi)}, Tempo medio per el: {np.mean(tempi)/len(legit_testing_list)} \n")  
    '''
    # Filtro base


    # FFNN
    for par in [8, 16, 64]:
        ffnn, model_size = create_train_FFNN(LBF_X_train, LBF_y_train, len(v), par, 0.001, "risultati")
        probs0, probs1 = FFNN.get_classifier_probs(ffnn, LBF_X_train, LBF_y_train)     
        false_negs, taus, _ = analysis.tau_analysis(probs0, probs1, fprs, LBF_ratios, phishing_list)
        BF_backups = analysis.create_BFsbackup(fprs, LBF_ratios, false_negs)
            
        tempi_LBF = pd.DataFrame(dtype = float)
        for _ in range(5):
            start_time = time.time()
            prediction = ffnn.predict(LBF_X_test) 
            end_time = time.time()
            total_time = (end_time - start_time)/len(legit_testing_list)
            print(f"Tempo predizione: {total_time}")
            _, _, tempo = analysis.LBF_empirical_analysis(prediction, legit_testing_list, fprs, LBF_ratios, taus, BF_backups)
            tempi_LBF = pd.concat((tempi_LBF, (tempo.dropna() + total_time)))

        false_negs, taus, _ = analysis.tau_analysis(probs0, probs1, fprs, SLBF_ratios, phishing_list)
        SLBFs = analysis.create_SLBF_filters(fprs, SLBF_ratios, false_negs, phishing_list)
        tempi_SLBF = pd.DataFrame(dtype = float)
        for _ in range(5):
            start_time = time.time()
            prediction = ffnn.predict(LBF_X_test) 
            end_time = time.time()
            total_time = (end_time - start_time)/len(legit_testing_list)
            print(f"Tempo predizione: {total_time}")
            _, _, _, _, tempo = analysis.SLBF_empirical_analysis(prediction, legit_testing_list, fprs, SLBF_ratios, taus, SLBFs)
            tempi_SLBF = pd.concat((tempi_SLBF, tempo.dropna() + total_time))

        with open(f'risultati/tempi/FFNN_{par}_tempiLBF.tex', 'a') as file:
            # LBF
            file.write(tempi_LBF.groupby(tempi_LBF.index).mean().to_latex(float_format="%.3E", bold_rows = True, position = 'H'))
            file.write(tempi_LBF.groupby(tempi_LBF.index).std().to_latex(float_format="%.3E", bold_rows = True, position = 'H'))
            file.write((tempi_LBF.groupby(tempi_LBF.index).mean()*len(legit_testing_list)).to_latex(bold_rows = True, float_format="%.3f", position = 'H'))

        with open(f'risultati/tempi/FFNN_{par}_tempiLBF.p', 'wb') as fp:
            pickle.dump(tempi_LBF.groupby(tempi_LBF.index).mean(), fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'risultati/tempi/FFNN_{par}_tempiLBF_std.p', 'wb') as fp:
            pickle.dump(tempi_LBF.groupby(tempi_LBF.index).std(), fp, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'risultati/tempi/FFNN_{par}_tempiSLBF.tex', 'a') as file:
            # SLBF
            file.write(tempi_SLBF.groupby(tempi_SLBF.index).mean().to_latex(float_format="%.3E", bold_rows = True, position = 'H'))
            file.write(tempi_SLBF.groupby(tempi_SLBF.index).std().to_latex(float_format="%.3E", bold_rows = True, position = 'H'))
            file.write((tempi_SLBF.groupby(tempi_SLBF.index).mean()*len(legit_testing_list)).to_latex(bold_rows = True, float_format="%.3f", position = 'H')) 

        with open(f'risultati/tempi/FFNN_{par}_tempiSLBF.p', 'wb') as fp:
            pickle.dump(tempi_SLBF.groupby(tempi_SLBF.index).mean(), fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'risultati/tempi/FFNN_{par}_tempiSLBF_std.p', 'wb') as fp:
            pickle.dump(tempi_SLBF.groupby(tempi_SLBF.index).std(), fp, protocol=pickle.HIGHEST_PROTOCOL)
    '''
    LBF_X_train, LBF_y_train, LBF_X_test, LBF_y_test, phishing_list, legit_testing_list = init.LBF_train_test_split(X, y, X_ff_bin_enc)
    for par in [8]:
        ffnn, model_size = create_train_FFNN(LBF_X_train, LBF_y_train, 210, par, 0.001, "risultati")
        probs0, probs1 = FFNN.get_classifier_probs(ffnn, LBF_X_train, LBF_y_train)

        false_negs, taus, _ = analysis.tau_analysis(probs0, probs1, fprs, LBF_ratios, phishing_list)
        BF_backups = analysis.create_BFsbackup(fprs, LBF_ratios, false_negs)
        tempi_LBF = pd.DataFrame(dtype = float)
        for _ in range(5):
            start_time = time.time()
            prediction = ffnn.predict(LBF_X_test) 
            end_time = time.time()
            total_time = (end_time - start_time)/len(legit_testing_list)

            print(f"Tempo predizione: {total_time}")
            _, _, tempo = analysis.LBF_empirical_analysis(prediction, legit_testing_list, fprs, LBF_ratios, taus, BF_backups)
            tempi_LBF = pd.concat((tempi_LBF, tempo.dropna() + total_time))

        false_negs, taus, _ = analysis.tau_analysis(probs0, probs1, fprs, SLBF_ratios, phishing_list)
        SLBFs = analysis.create_SLBF_filters(fprs, SLBF_ratios, false_negs, phishing_list)
        tempi_SLBF = pd.DataFrame(dtype = float)
        for _ in range(5):
            start_time = time.time()
            prediction = ffnn.predict(LBF_X_test) 
            end_time = time.time()
            total_time = (end_time - start_time)/len(legit_testing_list)

            print(f"Tempo predizione: {total_time}")
            _, _, _, _, tempo = analysis.SLBF_empirical_analysis(prediction, legit_testing_list, fprs, SLBF_ratios, taus, SLBFs)
            tempi_SLBF = pd.concat((tempi_SLBF, tempo.dropna() + total_time))

        with open(f'risultati/tempi/FFNN_{par}_tempiLBF.tex', 'a') as file:
            # LBF
            file.write(tempi_LBF.groupby(tempi_LBF.index).mean().to_latex(float_format="%.3E", bold_rows = True, position = 'H'))
            file.write(tempi_LBF.groupby(tempi_LBF.index).std().to_latex(float_format="%.3E", bold_rows = True, position = 'H'))
            file.write((tempi_LBF.groupby(tempi_LBF.index).mean()*len(legit_testing_list)).to_latex(bold_rows = True, float_format="%.3f", position = 'H'))

        with open(f'risultati/tempi/FFNN_{par}_tempiLBF.p', 'wb') as fp:
            pickle.dump(tempi_LBF.groupby(tempi_LBF.index).mean(), fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'risultati/tempi/FFNN_{par}_tempiLBF_std.p', 'wb') as fp:
            pickle.dump(tempi_LBF.groupby(tempi_LBF.index).std(), fp, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'risultati/tempi/FFNN_{par}_tempiSLBF.tex', 'a') as file:
            # SLBF
            file.write(tempi_SLBF.groupby(tempi_SLBF.index).mean().to_latex(float_format="%.3E", bold_rows = True, position = 'H'))
            file.write(tempi_SLBF.groupby(tempi_SLBF.index).std().to_latex(float_format="%.3E", bold_rows = True, position = 'H'))
            file.write((tempi_SLBF.groupby(tempi_SLBF.index).mean()*len(legit_testing_list)).to_latex(bold_rows = True, float_format="%.3f", position = 'H'))

        with open(f'risultati/tempi/FFNN_{par}_tempiSLBF.p', 'wb') as fp:
            pickle.dump(tempi_SLBF.groupby(tempi_SLBF.index).mean(), fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'risultati/tempi/FFNN_{par}_tempiSLBF_std.p', 'wb') as fp:
            pickle.dump(tempi_SLBF.groupby(tempi_SLBF.index).std(), fp, protocol=pickle.HIGHEST_PROTOCOL)

    # RNN
    LBF_X_train, LBF_y_train, LBF_X_test, LBF_y_test, phishing_list, legit_testing_list = init.LBF_train_test_split(X, y, X_rnn_enc)
    for par in [16, 8, 4]:
        rnn = R.RNN(emb_size = 5, h_size = par, layers = 1).to(device) 
        trainRNN.train(rnn, torch.tensor(LBF_X_train), torch.tensor(LBF_y_train), optimizer = torch.optim.Adamax(rnn.parameters()), device = device)
        model_size = R.model_size(rnn, f"risultati/modelli/RNN_{par}.p", use_pickle = True)
        rnn = R.load_pickle_model(f"risultati/modelli/RNN_{par}.p", h_size = par)
        probs1, probs0 = trainRNN.get_classifier_probs(rnn, torch.tensor(LBF_X_train), torch.tensor(LBF_y_train))

        false_negs, taus, _ = analysis.tau_analysis(probs0, probs1, fprs, LBF_ratios, phishing_list)
        BF_backups = analysis.create_BFsbackup(fprs, LBF_ratios, false_negs)
        test_set = torch.tensor(LBF_X_test).to(device)

        tempi_LBF = pd.DataFrame(dtype = float)
        for _ in range(5):
            start_time = time.time()       
            y_hat, _ = rnn(test_set)     
            prediction = torch.sigmoid(y_hat[:,:,1])[:,149].squeeze().detach().cpu().numpy()
            end_time = time.time()
            total_time = (end_time - start_time)/len(legit_testing_list)
            print(f"Tempo predizione: {total_time}")
            _, _, tempo = analysis.LBF_empirical_analysis(prediction, legit_testing_list, fprs, LBF_ratios, taus, BF_backups)
            tempi_LBF = pd.concat((tempi_LBF, tempo.dropna() + total_time))

        false_negs, taus, _ = analysis.tau_analysis(probs0, probs1, fprs, SLBF_ratios, phishing_list)
        SLBFs = analysis.create_SLBF_filters(fprs, SLBF_ratios, false_negs, phishing_list)

        tempi_SLBF = pd.DataFrame(dtype = float)
        for _ in range(5):
            start_time = time.time()            
            y_hat, _ = rnn(test_set)
            prediction = torch.sigmoid(y_hat[:,:,1])[:,149].squeeze().detach().cpu().numpy()
            end_time = time.time()
            total_time = (end_time - start_time)/len(legit_testing_list)
            print(f"Tempo predizione: {total_time}")
            _, _, _, _, tempo = analysis.SLBF_empirical_analysis(prediction, legit_testing_list, fprs, SLBF_ratios, taus, SLBFs)
            tempi_SLBF = pd.concat((tempi_SLBF, tempo.dropna() + total_time))

        with open(f'risultati/tempi/RNN_{par}_tempiLBF.tex', 'a') as file:
            # LBF
            file.write(tempi_LBF.groupby(tempi_LBF.index).mean().to_latex(float_format="%.3E", bold_rows = True, position = 'H'))
            file.write(tempi_LBF.groupby(tempi_LBF.index).std().to_latex(float_format="%.3E", bold_rows = True, position = 'H'))
            file.write((tempi_LBF.groupby(tempi_LBF.index).mean()*len(legit_testing_list)).to_latex(bold_rows = True, float_format="%.3f", position = 'H'))

        with open(f'risultati/tempi/RNN_{par}_tempiLBF.p', 'wb') as fp:
            pickle.dump(tempi_LBF.groupby(tempi_LBF.index).mean(), fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'risultati/tempi/RNN_{par}_tempiLBF_std.p', 'wb') as fp:
            pickle.dump(tempi_LBF.groupby(tempi_LBF.index).std(), fp, protocol=pickle.HIGHEST_PROTOCOL)

        with open(f'risultati/tempi/RNN_{par}_tempiSLBF.tex', 'a') as file:
            # SLBF
            file.write(tempi_SLBF.groupby(tempi_SLBF.index).mean().to_latex(float_format="%.3E", bold_rows = True, position = 'H'))
            file.write(tempi_SLBF.groupby(tempi_SLBF.index).std().to_latex(float_format="%.3E", bold_rows = True, position = 'H'))
            file.write((tempi_SLBF.groupby(tempi_SLBF.index).mean()*len(legit_testing_list)).to_latex(bold_rows = True, float_format="%.3f", position = 'H')) 

        with open(f'risultati/tempi/RNN_{par}_tempiSLBF.p', 'wb') as fp:
            pickle.dump(tempi_SLBF.groupby(tempi_SLBF.index).mean(), fp, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'risultati/tempi/RNN_{par}_tempiSLBF_std.p', 'wb') as fp:
            pickle.dump(tempi_SLBF.groupby(tempi_SLBF.index).std(), fp, protocol=pickle.HIGHEST_PROTOCOL)
    '''

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("fprs", type = float, nargs = "?", default = [0.001, 0.005, 0.01, 0.02], help = "Lista di false positive rate su cui testare le strutture")
    parse.add_argument("lbfratio", type = float, nargs = "?", default = [.1 * i for i in range(1, 10)], help = "Lista di fpr ratio usati per l'analisi di tau su LBF")
    parse.add_argument("slbfratio" , type = float, nargs = "?", default = [1. * i for i in range(1, 11)], help = "Lista di fpr ratio usati per l'analisi di tau su SLBF")
    args = parse.parse_args()

    main(args)