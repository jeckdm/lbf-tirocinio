import init 
import RNN 
import trainRNN 
import torch.nn as nn
import numpy as np
import time
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def classifier_evaluation(model, X_test, y_test, batchsize):
    start = time.time()
    predictions, targets = RNN.get_predictions(model, X_test, y_test, batchsize)
    end = time.time()
    execution_time = end-start
    print(execution_time)
    RNN_score = classification_report(targets, predictions, output_dict=True, target_names=['Legitimate', 'Phishing'], digits=4)

    return pd.DataFrame(RNN_score), execution_time

def main():
    # Fisso i parametri del classificatore pari a quelli utilizzati nell'articolo
    h_sizes = [16,8,4]
    emb_size = 5
    batchsize = 256
    criterion = nn.CrossEntropyLoss()
    # Carico dataset
    X, y = init.load_data(verbose=False)
    # Codifico gli URL, risultato é un tensor   
    X_encoded, y_encoded = init.map_to_number(X, y) 

    for t in [0.5, 0.67, 0.8]:
        print("Train size ", t)
        result = {16: [], 8: [], 4: []}
        total_execution_time = {16: 0, 8: 0, 4: 0}

        for it in range(5):
            # Suddivisione in training e testing
            X_train, X_test, y_train, y_test = train_test_split(X_encoded, y_encoded, train_size = t) 
            print(f"Train set: {X_train.shape} {y_train.shape}, Test set: {X_test.shape}")
            # Addestro il classificatore
            trainRNN.train(X_train, y_train, criterion, h_sizes, emb_size, batchsize)
            models = trainRNN.load_eval(X_test, y_test, criterion, h_sizes ,emb_size, batchsize)
            # Calcolo della matrice di confusione
            for model, h_size in zip(models.values(), h_sizes):
                df, exec_time = classifier_evaluation(model, X_test, y_test, batchsize)
                result[h_size].append(df)
                total_execution_time[h_size] += exec_time

        for key,val,total_time in zip(result.keys(), result.values(), total_execution_time.values()):
            total_df = pd.concat(val).groupby(level=0)
            with open((f"train-size{t}_{key}dim _result.txt"), "a") as myfile:
                myfile.write(total_df.mean().to_latex(position = "H", float_format="%.3f", caption="Risultati medi su 5 esecuzioni"))
                myfile.write(total_df.min().to_latex(position = "H", float_format="%.3f", caption="Risultati minimi su 5 esecuzioni"))
                myfile.write(total_df.max().to_latex(position = "H", float_format="%.3f", caption="Risultati massimi su 5 esecuzioni"))
                myfile.write(total_df.std().to_latex(position = "H", float_format="%.3f", caption="Dev.St su 5 esecuzioni"))
                myfile.write(f"\nTempo previsioni medio:\n{total_time/5:.3f} s")

if __name__ == "__main__":
    main()