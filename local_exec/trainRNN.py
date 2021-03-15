import numpy as np
import torch
import torch.nn as nn
import os
import time
import torch.nn.functional as F
import RNN as R
from RNN import val,train,RNN,make_batch_train,make_batch_test
#paramaters RNN
emb_size=5
h_sizes = [16,8,4]
layers = 1
criterion = nn.CrossEntropyLoss()


def give_params():
  return emb_size,h_sizes

def train(X_train,y_train,loc,device):
  models = {}
  for i in range(3):
    h_size = h_sizes[i]
    # Create model, loss function, optimizer
    models[i] = RNN(emb_size=emb_size, h_size=h_size, layers=layers).to(device)
    optimizer = torch.optim.Adamax(models[i].parameters())

    # Train and validate
    start = time.time()
    for epoch in range(30):
        train_loss = R.train(models[i],X_train,y_train,device,optimizer,criterion)
        val_acc, val_loss = val(models[i],X_train,y_train,device,criterion)
        if(epoch%10 == 0):
          print('[E{:4d}] Loss: {:.4f} | Acc: {:.4f}'.format(epoch, val_loss, val_acc))
    end = time.time()
    print(end-start)
    torch.save(models[i].state_dict(), loc+"RNN_emb"+str(emb_size)+"_hid"+str(h_size))  

def load(X_test,y_test,loc,device):
  models = {}
  model_sizes = {}
  for i,h_size in enumerate(h_sizes):
    print("hidden size", h_size)
    model_sizes[i] = os.path.getsize(loc+"RNN_emb"+str(emb_size)+"_hid"+str(h_size))
    print("model size (bytes)", model_sizes[i])
    models[i] = RNN(emb_size=emb_size, h_size=h_size, layers=layers).to(device)
    models[i].load_state_dict(torch.load(loc+"RNN_emb"+str(emb_size)+"_hid"+str(h_size)))
    models[i].eval()
    print(val(models[i], X_test,y_test,device,criterion))
    
    avg_time = 0
    for t in range(100):
      x,y = make_batch_test(X_test, y_test,1,device)
      start = time.time()
      models[i](x)
      total = time.time()-start
      avg_time += total
    avg_time /= 100
    print("time to evaluate", avg_time)
    print("="*30)
    return models

def get_classifier_probs(model,X_train,y_train,device):
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
    probs1 += list(ps[y==1].squeeze().detach().cpu().numpy())
    probs0 += list(ps[y==0].squeeze().detach().cpu().numpy())
  return probs1, probs0