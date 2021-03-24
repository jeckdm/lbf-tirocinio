# Libraries
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import init
# Parametri globali
import config

device = config.device
criterion= config.criterion

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

def val(model, X_t, y_t):
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
            x, y = make_batch_test(X_t, y_t, 256)
            y_hat, _ = model(x)
            #optimizer.zero_grad()
            preds = y_hat.max(dim=2)[1][:,149]
            preds_eq = preds.eq(y)
            total_right += preds_eq.sum().item()
            total += preds_eq.numel()
            loss = criterion(y_hat[:,149].view(-1,2), y.view(-1).long()).detach().item()
            total_loss += loss

    return total_right / total, total_loss / 50

def train(model,X_train,y_train,optimizer,criterion):
    '''
    Addestra model sul dataset (X_train, y_train).
    '''

    model.train()
    total_loss = 0
    for batch in range(50):
        x, y = make_batch_train(X_train, y_train, 256)
        y_hat, _ = model(x)
        optimizer.zero_grad()
        y_hat = y_hat.view(-1,2)
        y = y.view(-1).long()
        loss = criterion(y_hat, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / 50


def make_batch_train(X_t, y_t, B):
    '''
    Ritorna batch di training di dimensione B del dataset in input.
    '''

    sample = torch.randint(0, X_t.shape[0], [B]).long()
    batch_X0 = torch.stack([X_t[s] for s in sample])
    batch_X = batch_X0.to(device)
    batch_y0 = torch.stack([y_t[s]*torch.ones(len(X_t[s])).long() for s in sample]).to(device) # Moltiplicazione utile per calcolare funzione di errore
    batch_y = batch_y0.to(device)

    return batch_X, batch_y
  
def make_batch_test(X_t, y_t, B):
    '''
    Ritorna batch di testing di dimensione B del dataset in input.
    '''

    sample = torch.randint(0, X_t.shape[0], [B]).long()
    batch_X0 = torch.stack([X_t[s] for s in sample])
    batch_X = batch_X0.to(device)
    batch_y0 = torch.stack([y_t[s] for s in sample]).to(device)
    batch_y = batch_y0.to(device)

    return batch_X, batch_y


'''
- aggiunti commenti
- messo device come globale

Davide
-messo criterion come globale
'''