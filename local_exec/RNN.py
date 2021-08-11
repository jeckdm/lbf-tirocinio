# Libraries
import pickle
from tensorflow.python.eager.context import device
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import sys
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

class RNN(nn.Module):
    def __init__(self, input_size=150, output_size=2, emb_size=128, h_size=128, layers=1, dropout=0.3):
        super(RNN, self).__init__()
        self.embedding = nn.Embedding(input_size, emb_size)
        self.rnn = nn.GRU(emb_size, h_size, num_layers=layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(h_size, output_size)
        
    def forward(self, x, h=None):
        x = self.embedding(x)
        x, h = self.rnn(x, h)
        x = self.linear(x)
        return x, h

def val(model, X_t, y_t, criterion, batch_size, device):
    '''
    Valuta model sul dataset (X_t, y_t).
    Ritorna rapporto tra previsioni corrette / totale e loss su ogni batch
    '''

    model.eval()
    with torch.no_grad():
        total = 0
        total_right = 0
        total_loss = 0
        for batch in range(50):
            x, y = make_batch_test(X_t, y_t, batch_size, device)
            y_hat, _ = model(x)
            preds = y_hat.max(dim=2)[1][:,149]
            preds_eq = preds.eq(y)
            total_right += preds_eq.sum().item()
            total += preds_eq.numel()
            loss = criterion(y_hat[:,149].view(-1,2), y.view(-1).long()).detach().item()
            total_loss += loss

    return total_right / total, total_loss / 50

def get_predictions(model, X_test, y_test, device = torch.device('cpu')):
    model.eval()
    with torch.no_grad():
        # Calcolo gli output del modello sul sample
        y_hat, _ = model(X_test.to(device)) 
        # Predizione pari allo score piú altro tra i 2 output
        preds = y_hat.max(dim=2)[1][:,149]  
        # Aggiungo il risultato a predictions e targets
        predictions = preds.detach().cpu().numpy() # Riporto in memoria CPU e trasformo in array numpy
        targets = y_test.detach().cpu().numpy()

    return predictions, targets

def train(model, X_train, y_train, optimizer, criterion, batch_size, device = torch.device('cpu')):
    '''
    Addestra model sul dataset (X_train, y_train).
    '''

    model.train()
    total_loss = 0
    for batch in range(50):
        x, y = make_batch_train(X_train, y_train, batch_size, device)
        y_hat, _ = model(x)
        optimizer.zero_grad()
        y_hat = y_hat.view(-1,2)
        y = y.view(-1).long()
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
    total_loss += loss.item()
    return total_loss / 50


def make_batch_train(X_t, y_t, B, device):
    '''
    Ritorna batch di training di dimensione B del dataset in input.
    '''

    sample = torch.randint(0, X_t.shape[0], [B]).long()
    batch_X0 = torch.stack([X_t[s] for s in sample])
    batch_X = batch_X0.to(device)
    batch_y0 = torch.stack([y_t[s]*torch.ones(len(X_t[s])).long() for s in sample]).to(device) # Moltiplicazione utile per calcolare funzione di errore
    batch_y = batch_y0.to(device)

    return batch_X, batch_y
  
def make_batch_test(X_t, y_t, B, device):
    '''
    Ritorna batch di testing di dimensione B del dataset in input.
    '''

    sample = torch.randint(0, X_t.shape[0], [B]).long()
    batch_X0 = torch.stack([X_t[s] for s in sample])
    batch_X = batch_X0.to(device)
    batch_y0 = torch.stack([y_t[s] for s in sample]).to(device)
    batch_y = batch_y0.to(device)

    return batch_X, batch_y


def score_report(model, X_test, y_test):
    predictions, targets = get_predictions(model, torch.tensor(X_test), torch.tensor(y_test))
    print(confusion_matrix(targets, predictions))
    RNN_score = classification_report(targets, predictions, output_dict=True)

    return RNN_score

def model_size(model, location = None, use_pickle = True, verbose = True):
    weight_dict = model.state_dict()

    if use_pickle:
        with open(location, "wb") as file:
            pickle.dump(weight_dict, file)
    else:
        torch.save(weight_dict, location)

    size = os.path.getsize(location)

    if verbose: 
        print(f"Dimensione oggetto in memoria: {sys.getsizeof(weight_dict)}")
        print(f"Dimensione oggetto su disco: {size}")

    return size

def load_pickle_model(location, input_size = 150, output_size = 2, emb_size = 5, h_size = 16, layers = 1, dropout = 0.3):
    """ Ritorna RNN, inizializzata con parametri in input, con i pesi presenti nel file pickle in location"""

    model = RNN(input_size, output_size, emb_size, h_size, layers, dropout)
    with open(location, "rb") as file:
        model.load_state_dict(pickle.load(file))

    return model