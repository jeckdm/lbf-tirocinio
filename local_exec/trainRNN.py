import numpy as np
import torch
import torch.nn as nn
import time
import torch.nn.functional as F
import RNN as R
from RNN import val,train,RNN,make_batch_train
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