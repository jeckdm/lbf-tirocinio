import torch.nn as nn
import time
import RNN as R

def train(model, X_train, y_train, device, optimizer, criterion = nn.CrossEntropyLoss(), batch_size = 256):
  """ Train modificato per lavorare solamente su un modello, creata per non modifcare funzione originale """
  # Train and validate
  for epoch in range(30):
      _ = R.train(model, X_train, y_train, optimizer, criterion, batch_size)
      val_acc, val_loss = R.val(model, X_train, y_train, criterion, batch_size)
      if(epoch%10 == 0):
        print('[E{:4d}] Loss: {:.4f} | Acc: {:.4f}'.format(epoch, val_loss, val_acc))

  return model