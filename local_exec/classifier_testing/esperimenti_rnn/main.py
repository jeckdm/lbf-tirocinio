import init 
import RNN 
import trainRNN 
import torch.nn as nn
import numpy as np
import time
import torch
import pandas as pd
import argparse
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from collections import Counter

def classifier_evaluation(model, X_test, y_test, batchsize):
    start = time.time()
    predictions, targets = RNN.get_predictions(model, torch.tensor(X_test), torch.tensor(y_test), batchsize)
    end = time.time()
    print(confusion_matrix(targets, predictions), end-start)
    RNN_score = classification_report(targets, predictions, output_dict=True, target_names=['Legitimate', 'Phishing'], digits=4)

    return pd.DataFrame(RNN_score)

def formatter(dfs):
    df_out = pd.DataFrame(columns = ['16 Dimensioni', '8 Dimensioni', '4 Dimensioni'], index = ['f1-score', 'precision', 'recall', 'accuracy'])

    for i, df in enumerate(dfs):
        accuracy = df['accuracy'][0]
        scores = df['Phishing'][:3].tolist()
        df_out.iloc[:,i] = scores + [accuracy]

    return df_out

def encode(X, y, d, char_cutoff = 150):
    X_encoded = [[d[l] if l in d else 0 for l in url] for url in X]
    X_encoded = [[l for l in url[:min([len(url),char_cutoff])]] + [0 for l in range(char_cutoff-len(url))] for url in X_encoded]
    y_encoded = y

    return np.array(X_encoded), np.array(y_encoded)

def main(args):
    # Fisso i parametri del classificatore pari a quelli utilizzati nell'articolo
    h_sizes = [16,8,4]
    emb_size = 5
    batchsize = 256
    criterion = nn.CrossEntropyLoss()
    # Carico dataset
    X, y = init.load_data(verbose=False)

    for train_size in args.trainsize:
        print("Train-size: ", train_size)
        for ratio in args.ratiolp:
            result = {16: [], 8: [], 4: []}
            print("Ratio Phishing/Legit: ", ratio)
            for i in range(5):
                # Mantengo fisso testing fisso e faccio undersampling in training
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = train_size, stratify = y)

                # Ricavo dizioniario per la codifica utilizzando solo insieme di train
                encoding_dict = init.map_to_number(X_train, y_train)
                # Codifico train test
                X_train_encoded, y_train_encoded = encode(X_train, y_train, encoding_dict)
                X_test_encoded, y_test_encoded = encode(X_test, y_test, encoding_dict)

                # Bilanciamento dataset
                X_train_encoded, y_train_encoded = init.undersample(X_train_encoded, y_train_encoded, ratio = ratio)
                
                # Addestro il classificatore
                trainRNN.train(torch.tensor(X_train_encoded), torch.tensor(y_train_encoded), criterion, h_sizes, emb_size, batchsize)
                models = trainRNN.load_eval(torch.tensor(X_test_encoded), torch.tensor(y_test_encoded), criterion, h_sizes ,emb_size, batchsize)
                # Calcolo prestazioni
                for model, h_size in zip(models.values(), h_sizes):
                    print(model)
                    df = classifier_evaluation(model, X_test_encoded, y_test_encoded, batchsize)
                    result[h_size].append(df)

            df_mean = formatter([pd.concat(result[key]).groupby(level=0).mean() for key in result.keys()])
            df_std = formatter([pd.concat(result[key]).groupby(level=0).std() for key in result.keys()])
            with open((f"train-size{train_size}_undersampling{ratio}.txt"), "a") as myfile:
                myfile.write(df_mean.to_latex(position = "H", float_format="%.3f", caption="Risultati medi su 5 esecuzioni"))
                myfile.write(df_std.to_latex(position = "H", float_format="%.3f", caption="Std su 5 esecuzioni"))


if __name__ == "__main__":
    parse = argparse.ArgumentParser(description="Argument for analysis")
    parse.add_argument("--ratiolp", "-r", type=float, nargs="+", default = [None])
    parse.add_argument("--trainsize", "-t", type=float, nargs="+", default = [0.67])

    args=parse.parse_args()

    main(args)