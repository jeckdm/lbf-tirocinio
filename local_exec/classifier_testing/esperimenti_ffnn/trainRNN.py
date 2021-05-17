import torch
import torch.nn as nn
import os
import time
import RNN as R

def train(model, X_train, y_train, device, h_size, optimizer, criterion = nn.CrossEntropyLoss(), emb_size = 5, batch_size = 256, layers = 1):
    # Train and validate
    start = time.time()
    for epoch in range(30):
        train_loss = R.train(model,X_train,y_train,optimizer,criterion,batch_size, device)
        val_acc, val_loss = R.val(model,X_train,y_train,criterion,batch_size, device)
        if(epoch%10 == 0):
          print('[E{:4d}] Loss: {:.4f} | Acc: {:.4f}'.format(epoch, val_loss, val_acc))
    end = time.time()
    print(end-start)

    return model

def get_classifier_probs(model, X_train, y_train):
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