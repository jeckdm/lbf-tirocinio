# Estrae solo URL da dataset

import numpy as np

# Numero di entry
n  = 100
# Legit o phishing, 'L' o 'P'
s = 'L'
# Path della directory principale del dataset
path ='pathdesiderato'

# Genero nomi delle directory del tipo 'L00001', 'L00002',...
num = ['%d' % i for i in range(n)]
dirs = [s + '0'*(5 - len(num[i])) + str(i+1) for i in range (n)]

data = np.array([])

for dirname in dirs:
    with open(path + dirname + '/URL/URL.txt', 'r') as outfile:
        data = np.append(data, outfile.read())
        outfile.close()

np.save('data.npy', data)
