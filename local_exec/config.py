import torch
import torch.nn as nn 

# Device e location
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
loc_data1 = "data/dataset1/"
loc_data2 = "data/dataset2/"
loc_data3 = "data/dataset3/"
loc_nn = "test/simulations/"
loc_plots = "test/plots/"

# Parametri RNN
emb_size= 5
h_sizes = [16,8,4]
layers = 1