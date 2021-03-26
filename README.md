# lbf-tirocinio

## conversione da archivio a dataset .npy:<br>
utilizzare URL_script.py -> inserire il path dell'archivio al posto di 'pathdesiderato'

## impostare enviroment esecuzione
per eseguire il notebook utilizzeremo un virtual enviroment. Per creare il virtual env. assicurarsi di aver installato anaconda (https://www.anaconda.com/products/individual) ed eseguire il seguente script nella home directory del git:
```bash
conda env create -f env_settings/enviroment.yml 
```
in seguito attivare l'env ed eseguire il seguente comando:

```bash
ldconfig -p|grep cudart.so|sed -e 's/.*\.\([0-9]*\)\.\([0-9]*\)$/cu\1\2/'
```
nel caso di utenti linux, oppure:
```bash
conda env create -f env_settings/enviroment_win.yml 
```
nel caso di utenti windows.

prima di eseguire il codice assicurarsi di aver attivato l'env  
```bash 
conda activate LBF-env
``` 

dopo l'esecuzione si consiglia di disattivare l'env 
```bash 
conda deactivate
```

## esecuzione notebook: 
per eseguire il notebook assicurarsi le path da cui vengono caricati i file coincidano con quelle in cui avete scaricato il dataset. Al momento i due dataset utilizzati sono in 'small-data' e le path del notebook cercano i file lì.

## esecuzione codice:
per eseguire direttamente il codice (senza utilizzare il notebook) eseguire main.py situato nella cartella local_excec. la maggior parte dei parametri possono essere direttamente impostati cambiando i valori inseriti nel file config.py (size embedding, dimensioni GRU, fpr/fpr ratio utilizzate per la creazione di grafici...)
