import numpy as np
import sys
import os
import pandas as pd
import pickle
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy
from cycler import cycler

def create_sequential(input_size, activation = 'relu', hidden_layer_size = 32, loss = BinaryCrossentropy(), optimizer = 'adam', metrics = ['accuracy'], learning_rate = 0.001):
    """ 
    Costruisce un modello Feed Forward con la seguente struttura
    
    Strato input:           ( , 82)
    ------------------------
    Strato Dense Hidden:    Input   ( ,82)                       Output  ( , hidden_layer_size)
    ------------------------
    Strato Dense output:    Input   ( , hidden_layer_size)       Output  ( , 1)
    """
    model = Sequential()

    model.add(Dense(hidden_layer_size, activation = activation, input_shape = input_size))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer = Adam(learning_rate = learning_rate),
        loss = loss,
        metrics = metrics)

    return model

def train(model, X_train, y_train, epochs = 10, batch_size = 128, validation_split = 0.2, verbose = 1, cbs = None):
    """ Addestra il modello su X_train, y_train """

    history = model.fit(
        X_train, y_train,
        batch_size = batch_size,
        epochs = epochs,
        verbose = verbose,
        validation_split = validation_split,
        callbacks = cbs
    )

    return history

def evaluate(model, X_test, y_test, verbose = True, result_loc = None):
    """ Valuta il modello su X_test, y_test """

    predictions = (model.predict(X_test) > 0.5).astype("int32")
    targets = y_test

    scores = classification_report(targets, predictions, output_dict = True)

    if verbose:
        print(confusion_matrix(targets, predictions))
        print(scores)

    if result_loc is not None:
        scores = pd.DataFrame(scores).drop(columns = ['macro avg', 'weighted avg'], index = 'support').round(3)
        with open(result_loc, 'a') as file:
            file.write(scores.to_latex(position = "H"))

    return scores

def model_size(model, location = None, verbose = True, use_pickle = True):
    """ Ritorna la dimensione del modello, calcolata valutando la dimensione del file numpy dei pesi """
    
    weights = model.get_weights()

    # Salvo il file e ne calcolo la dimensione
    if use_pickle:
        with open(location, "wb") as file:
            pickle.dump(weights, file)
    else:
        np.save(location, weights)

    size = os.path.getsize(location)

    if verbose: 
        print(f"Dimensione oggetto in memoria: {sys.getsizeof(weights)}")
        print(f"Dimensione oggetto su disco: {size}")

    return size

def load_pickle_model(location, input_size, activation = 'relu', hidden_layer_size = 32, loss = BinaryCrossentropy(), optimizer = 'adam', metrics = ['accuracy'], learning_rate = 0.001):
    """ Inizializza FFNN con iperparametri in input e pesi caricati dal file pickle indicati in location """

    model = create_sequential(input_size, activation, hidden_layer_size, loss, optimizer, metrics, learning_rate)
    with open(location, "rb") as file:
        # Ricavo array dei pesi
        weights = pickle.load(file)
        # Carico i pesi
        model.set_weights(weights)

    return model

def save_losses_plot(history, name, colors):
    """ Salva il grafico delle curve di apprendimento ricavate dalla lista di history in ingresso, len(colors) == len(history), history devono essere passate in ordine crescente"""

    plt.figure(figsize=(8, 6))
    plt.rc('axes', prop_cycle=(cycler('color', colors) * cycler('linestyle', ['-', '--']) ))

    for count, el in enumerate(history, 1):
        hist = el.history
        plt.plot(hist['loss'], label = f'Fold{count} train loss')
        if 'val_loss' in hist:
            plt.plot(hist['val_loss'], label = f'Fold{count} val loss')

    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid(True)
    plt.legend(loc='best')

    plt.savefig(f"testing/risultati/4/{name}.png")

def get_classifier_probs(model, X, y):
    """ Restituisce liste contenenti rispettivamente le predizioni del classificatore per la classe 0 e la classe 1 """
    # Ho già una sigmoide come ultimo strato del modello?
    predictions = model.predict(X)

    probs1 = predictions[y == 1]
    probs0 = predictions[y == 0]

    return probs0, probs1






