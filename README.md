# lbf-tirocinio

## Download dataset2
Per scaricare il secondo dataset eseguire lo script `getdataset.py` presente nella cartella data/dataset2

## Enviroment
Per eseguire il codice è necessario avere diversi package installabili tramite conda utilizzando l'enviroment presente nella cartella env_settings. Per creare il virtual env. assicurarsi di aver installato anaconda (https://www.anaconda.com/products/individual) ed eseguire il seguente script nella home directory del git:
```bash
conda env create -f env_settings/env.yml 
```
In seguito attivare l'env ed eseguire il seguente comando:
```bash
conda update -all
```
In modo da aggiornare tutti i package per il sistema operativo utilizzato.

## Esecuzione codice:
Per replicare i risultati ottenuti è necessario eseguire lo script `main.py` presente nella cartella local_exec, lo script necessita di due argomenti obbligatori da linea di comando: rispettivamente la cartella in cui verranno salvati i risultati e la cartella da cui verranno prelevati gli URL.
