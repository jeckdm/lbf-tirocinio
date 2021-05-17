import keras
import helpers
import numpy as np
import nn_sequential as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.utils.vis_utils import plot_model
from sklearn.model_selection import StratifiedKFold

def main():
    # Carico dataset
    X, y = helpers.load_data()

    # Codifico (A priori non aggiungo informazioni)
    X_encoded, y_encoded, vocabulary = helpers.encode(X, y)

    # Suddivido in training-testing
    X_train_encoded, X_test_encoded, y_train_encoded, y_test_encoded = train_test_split(X_encoded, y_encoded, train_size = 0.67)
    print(np.shape(X_train_encoded), np.shape(X_test_encoded))

    # Bilanciamento
    X_train_encoded, y_train_encoded = helpers.undersample(X_train_encoded, y_train_encoded, ratio = 0.2)
    print(np.shape(X_train_encoded))

    # Permutazioni casuali delle entry (Per validation)
    np.random.seed(0)
    random_permutation = np.random.permutation(len(X_train_encoded))
    X_train_encoded = X_train_encoded[random_permutation]
    y_train_encoded = y_train_encoded[random_permutation]

    # Creo modello
    model = nn.create_sequential(input_size = (len(vocabulary), ), hidden_layer_size = 128, activation = 'relu')
    print(model.summary())
    plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=True)

    # Training
    nn.train(model, X_train_encoded, y_train_encoded, validation_split = 0.2)

    # Score del classificatore sul testing
    scores = nn.evaluate(model, X_test_encoded, y_test_encoded, result_loc = "test.tex")
    
if __name__ == "__main__":
    main()