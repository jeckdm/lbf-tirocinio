import numpy as np
import sys
import os
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from keras.optimizers import Adam
from keras.losses import BinaryCrossentropy

def create_sequential(input_size, activation = 'relu', hidden_layer_size = 32, loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy'], learning_rate = 0.001):
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
        loss = BinaryCrossentropy(),
        metrics = ['accuracy'])

    return model

def train(model, X_train, y_train, epochs = 10, batch_size = 128, validation_split = 0.2, verbose = 1):
    """ Addestra il modello su X_train, y_train """

    history = model.fit(
        X_train, y_train,
        batch_size = batch_size,
        epochs = epochs,
        verbose = verbose,
        validation_split = validation_split
    )

    return history

def evaluate(model, X_test, y_test, verbose = True, result_loc = None):
    """ Valuta il modello su X_test, y_test """

    predictions = (model.predict(X_test) > 0.5).astype("int32")
    targets = y_test

    scores = classification_report(targets, predictions, output_dict = True)
    # scores = pd.DataFrame(scores).drop(columns = ['macro avg', 'weighted avg'], index = 'support').round(3)

    if verbose:
        print(confusion_matrix(targets, predictions))
        print(scores)

    if result_loc is not None:
        with open(result_loc, 'a') as file:
            file.write(scores.to_latex(position = "H"))

    return scores

def model_size(model, verbose = True):
    """ Ritorna la dimensione del modello, calcolata valutando la dimensione del file numpy dei pesi """
    
    weights = model.get_weights()
    # Salvo il file e ne calcolo la dimensione
    np.save("res", weights)
    size = os.path.getsize("res.npy")

    if verbose: 
        print(f"Dimensione oggetto in memoria: {sys.getsizeof(weights)}")
        print(f"Dimensione oggetto su disco: {size}")

    os.remove("res.npy")

    return size

def save_loss_plot(history, name):
    """ Salva il grafico riguardante la loss del modello addestrato in funzione al numero di epoch """

    fig = plt.figure(figsize=(8, 6))

    for fold, h in history.items():
        plt.plot(h.history['loss'])

    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.savefig(f"{name}.png")



