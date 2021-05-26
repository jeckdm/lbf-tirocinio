# Syspath
import sys
sys.path.append('.\local_exec')

# Librerie
import init
from classifier_testing.esperimenti_ffnn import FFNN
from classifier_testing.esperimenti_ffnn_bf import LBF

def main():
    # Carico dataset
    X, y = init.load_data()
    # Codifica con CountVect
    X_enc, y_enc, vocabulary = init.CV_encode(X, y)
    # Suddivisione in training e testing per la struttura LBF
    # Train: Legit/2 + Phish
    # Test: Legit/2
    LBF_X_train, LBF_y_train, LBF_X_test, LBF_y_test, LBF_training_list, LBF_testing_list = init.LBF_train_test_split(X, y, X_enc)
    # Creazione del modello
    model = FFNN.create_sequential(input_size = (len(vocabulary), ), hidden_layer_size = 30, learning_rate = 0.001)
    # Addestramento modello
    FFNN.train(model, LBF_X_train, LBF_y_train, epochs = 30, validation_split = None)
    # Valutazione del modello
    scores = FFNN.evaluate(model, LBF_X_test, LBF_y_test)

    probs0, probs1 = FFNN.get_classifier_probs(model, LBF_X_train, LBF_y_train)
    false_negs, tau = LBF.build_LBF_classifier(probs0, probs1, LBF_training_list, 0.02)
    BF_backup = LBF.build_LBF_backup(false_negs, 0.01, 0.001)
    probs0, probs1 = FFNN.get_classifier_probs(model, LBF_X_test, LBF_y_test)
    print(LBF.test_LBF(BF_backup, tau, LBF_testing_list, probs0))


if __name__ == "__main__":
    main()