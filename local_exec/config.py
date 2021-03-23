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
loc_data = "small_data/"
loc_nn = "test/simulations/"
loc_plots = "test/plots/"

# Parametri RNN
emb_size= 5
h_sizes = [24,16,8,4]
layers = 1
criterion = nn.CrossEntropyLoss()

# Parametri per testing e analisi di tau
fprs = [0.001,0.005,0.01,0.02]
fpr_ratios = [0.1*i for i in range(1,11)] # Ratio per LBF
fpr_ratios2 = [1.*i for i in range(1,11)] # Ratio per SLBF


''' 
rimosso import init 
aggiunti criterion ai parametri RNN
'''