import torch
import torch.nn as nn 


def GPU_init():                           
    '''
    Ritorna, se presente, il device cuda, altrimenti la cpu (con relativo accelerator).
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # accelerator = cuda_output[0] if torch.cuda.is_available() else 'cpu'
    print(f"Dispositivo: {device}")

    if device.type == 'cuda':
        print(f"Nome dispositivo {torch.cuda.get_device_name(0)}")

    return device

# Device e location
device = GPU_init()
loc_data1 = "data/dataset1/"
loc_data2 = "data/dataset2/"
loc_data3 = "data/dataset3/"
loc_nn = "test/simulations/"
loc_plots = "test/plots/"

# Parametri RNN
emb_size= 5
h_sizes = [4,8,16]
layers = 1
criterion = nn.CrossEntropyLoss()

# Parametri per testing e analisi di tau



''' 
rimosso import init 
aggiunti criterion ai parametri RNN
'''