import torch
import torch.nn as nn
import os
import time
import RNN as R
# Parametri globali
import config

# Rinomino parametri globali per comoditá
layers = config.layers
device = config.device


def train(X_train, y_train,criterion,h_sizes,emb_size,batch_size):
  '''
  Effettua l'addestramento di model sul dataset (X_train, y_train) utilizzando criterion come funzione di loss.
    '''
  for i,h_size in enumerate(h_sizes):
    models = {}
    # Create model, loss function, optimizer
    models[i] = R.RNN(emb_size=emb_size, h_size=h_size, layers=config.layers).to(device)
    optimizer = torch.optim.Adamax(models[i].parameters())
    # Train and validate
    start = time.time()
    for epoch in range(30):
        train_loss = R.train(models[i],X_train,y_train,optimizer,criterion,batch_size)
        val_acc, val_loss = R.val(models[i],X_train,y_train,criterion,batch_size)
        if(epoch%10 == 0):
          print('[E{:4d}] Loss: {:.4f} | Acc: {:.4f}'.format(epoch, val_loss, val_acc))
    end = time.time()
    torch.save(models[i].state_dict(), config.loc_nn+"RNN_emb"+str(emb_size)+"_hid"+str(h_size))  
    print(end-start)  

def load_eval(X_test, y_test,criterion,h_sizes,emb_size,batch_size):
  '''
  Carica i parametri delle RNN giá addestrate presenti in loc e ritorna una lista di RNN definite con gli  iperparametri contenuti in (h_sizes, emb_sizes, layers).

  Stampa inoltre i risultati di RNN.eval(models[i], X_test,y_test,criterion) su ognuna di queste RNN.
  '''

  models = {}
  model_sizes = {}

  for i,h_size in enumerate(h_sizes):
    print("hidden size", h_size)
    model_sizes[i] = os.path.getsize(config.loc_nn+"RNN_emb"+str(emb_size)+"_hid"+str(h_size))
    print("model size (bytes)", model_sizes[i])
    models[i] = R.RNN(emb_size=emb_size, h_size=h_size, layers=layers).to(device)
    models[i].load_state_dict(torch.load(config.loc_nn+"RNN_emb"+str(emb_size)+"_hid"+str(h_size)))
    # models[i].eval()  #in teoria superfluo
    print(R.val(models[i], X_test,y_test,criterion,batch_size))
    
    avg_time = 0
    for t in range(100):
      x,y = R.make_batch_test(X_test, y_test,1)
      start = time.time()
      models[i](x)
      total = time.time()-start
      avg_time += total
    avg_time /= 100
    print("time to evaluate", avg_time)
    print("="*30)

  return models

def get_classifier_probs(model,X_train,y_train):
  '''
  Ritorna le previsioni di model sul dataset (X_train, y_train).
  Le previsioni vengono ritornate come:
    probs1 = previsioni su URL phishing
    probs0 = previsioni su URL legit
  '''

  probs1 = []
  probs0 = []
  # Divisione in batch da 100
  for i in range(int(len(y_train)/100)+1):
    x0 = torch.stack([X_train[s] for s in range(100*i, min(100*(i+1), len(y_train)))])
    x = x0.to(device)
    y0 = torch.stack([y_train[s] for s in range(100*i, min(100*(i+1), len(y_train)))])
    y = y0.to(device)  
    y_hat, _ = model(x) 
    ps = torch.sigmoid(y_hat[:,:,1])[:,149]
    if(len(ps[y==1]) != 1):
      probs1 += list(ps[y==1].squeeze().detach().cpu().numpy()) 
    else:
      probs1 += list(ps[y==1].detach().cpu().numpy()) 
    if(len(ps[y==0]) != 1):
      probs0 += list(ps[y==0].squeeze().detach().cpu().numpy())
    else:
      probs0 += list(ps[y==0].detach().cpu().numpy())

  return probs1, probs0