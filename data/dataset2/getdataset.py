import wget
import zipfile
import pandas as pd
import numpy as np
import os

def download_zip(dest_directory):
    url= "https://drive.google.com/uc?export=download&id=1E-VOdDDbiC6HUzkadjxGIauUs4hwcyOx"
    zip_name= wget.download(url,"dataset.zip")
    zf = zipfile.ZipFile(zip_name)
    filename = zf.namelist()
    filename=str(filename.pop())
    zf.extractall(dest_directory)
    zf.close()
    os.remove(zip_name)
    return filename

def make_dataset(dest_directory,filename):
    print(filename)
    content= pd.read_csv(dest_directory+filename,sep=";")
    legit = content.loc[content['compromissionType']=='normal',"url"]
    phishing = content.loc[content['compromissionType']=='phishing',"url"]
    print (len(legit))
    print (len(phishing))
    legit_array = np.array(legit)
    phishing_array = np.array(phishing)
    np.save(dest_directory+"legitimate_URLs2.npy",legit_array)
    np.save(dest_directory+"phishing_URLs2.npy",phishing_array)
    print(phishing_array)
    os.remove(dest_directory+filename)
    
dest_directory="data/dataset2/"
make_dataset(dest_directory,download_zip(dest_directory))