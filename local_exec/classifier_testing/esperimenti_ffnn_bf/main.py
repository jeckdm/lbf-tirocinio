# Syspath
import sys

sys.path.append('.\local_exec')

# Librerie
import init
import time
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import RNN
from classifier_testing.esperimenti_ffnn import trainRNN
from classifier_testing.esperimenti_ffnn import FFNN
from classifier_testing.esperimenti_ffnn_bf import LBF

def train_FFNNs(models, X_train, y_train, X_test, y_test):
    """ Addestra FFNNs e ritorna Size | Train Time | Eval Time | Accuracy su Test """

    results = {'size': [], 'train_time': [], 'eval_time': [], 'test_accuracy': []}

    for idx in range(len(models)):
        # Addestramento modello
        train_start = time.time()
        FFNN.train(models[idx], X_train, y_train, epochs = 30, validation_split = None)
        train_end = time.time()
        # Valutazione del modello
        eval_start = time.time()
        accuracy = FFNN.evaluate(models[idx], X_test, y_test)['accuracy']
        eval_end = time.time()
        training_time = train_end - train_start
        eval_time = eval_end - eval_start
        size = FFNN.model_size(models[idx], location = "model.p")

        results['size'].append(size)
        results['train_time'].append(training_time)
        results['eval_time'].append(eval_time)
        results['test_accuracy'].append(accuracy)

    return results

def train_RNNs(models, X_train, y_train, X_test, y_test):
    """ Addestra RNNs e ritorna Size | Train Time | Eval Time | Accuracy su Test """

    results = {'size': [], 'train_time': [], 'eval_time': [], 'test_accuracy': []}

    for idx in range(len(models)):
        optimizer = torch.optim.Adamax(models[idx].parameters())
        # Addestramento modello
        train_start = time.time()
        trainRNN.train(models[idx], torch.tensor(X_train), torch.tensor(y_train), optimizer)
        train_end = time.time()
        # Valutazione del modello
        eval_start = time.time()
        accuracy = RNN.val(models[idx], torch.tensor(X_test), torch.tensor(y_test), criterion = nn.CrossEntropyLoss(), batch_size = 128)
        eval_end = time.time()
        training_time = train_end - train_start
        eval_time = eval_end - eval_start
        size = RNN.model_size(models[idx], location = "model2.p")

        results['size'].append(size)
        results['train_time'].append(training_time)
        results['eval_time'].append(eval_time)
        results['test_accuracy'].append(accuracy)
    
    return results

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Hyper param modelli
    FFNN_params = [(8, 0.01), (8, 0.001), (16, 0.01), (16, 0.001), (24, 0.01), (24, 0.001)]
    RNN_params = [16, 8, 4]
    # Carico dataset
    X, y = init.load_data()
    
    # Codifica con CountVect
    X_ff_enc, y_ff_enc, vocabulary = init.CV_encode(X, y)
    # Codifica RNN
    d = init.map_to_number(X)
    X_rnn_enc, y_rnn_enc = init.RNN_encode(X, y, d)

    # Dizionari modelli
    models_FFNN = {}
    models_RNN = {}

    # Creazione dei modelli
    for idx, (neuron, lr) in enumerate(FFNN_params):
        models_FFNN[idx] = FFNN.create_sequential(input_size = (len(vocabulary), ), hidden_layer_size = neuron, learning_rate = lr)
    for idx, hidden_layer in enumerate(RNN_params):
        models_RNN[idx] = RNN.RNN(emb_size = 5, h_size = hidden_layer, layers = 1).to(device)
        
    # Suddivisione in training e testing per la struttura LBF
    # Train: Legit/2 + Phish
    # Test: Legit/2
    # Suddivisione è la stessa per RNN e FFNN, solo con codifiche diverse
    LBF_X_train, LBF_y_train, LBF_X_test, LBF_y_test, LBF_training_list, LBF_testing_list = init.LBF_train_test_split(X, y, X_ff_enc)
    ff_results = train_FFNNs(models_FFNN, LBF_X_train, LBF_y_train, LBF_X_test, LBF_y_test)
    LBF_X_train, LBF_y_train, LBF_X_test, LBF_y_test, LBF_training_list, LBF_testing_list = init.LBF_train_test_split(X, y, X_rnn_enc)
    rnn_results = train_RNNs(models_RNN, LBF_X_train, LBF_y_train, LBF_X_test, LBF_y_test)

    # Unisco i risultati e salvo la tabella
    results = {key : ff_results[key] + rnn_results[key] for key in ff_results.keys()}
    results = pd.DataFrame.from_dict(results).round(3)

    # Salvo
    with open("local_exec/classifier_testing/risultati/tables/8_FFNN_BloomFilter/classifier_table.tex", "a") as file:
        file.write(results.to_latex(position = 'H'))
 


if __name__ == "__main__":
    main()