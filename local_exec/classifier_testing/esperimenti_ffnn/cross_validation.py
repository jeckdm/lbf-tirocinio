from itertools import count
import numpy as np
import torch
import init
import RNN as R
from classifier_testing.esperimenti_ffnn import trainRNN
from classifier_testing.esperimenti_ffnn import FFNN as ff
from classifier_testing.esperimenti_ffnn import helpers
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import load_model

def model_selection(X, y, params, undersample = False, epochs = 30, ratio = 0.2, outer_folds = 5, inner_folds = 5, njobs = None): 
    # Outer cv
    outer_cv = StratifiedKFold(n_splits = outer_folds)

    # Salvo risultati per ogni configurazione di parametri
    model_results = {f'fold{i}' : {'f1' : 0, 'precision' : 0, 'recall' : 0, 'accuracy' : 0} for i in range(outer_folds)}
    param_results = {f'fold{i}' : {} for i in range(outer_folds)}
    best_models = {f'fold{i}' : None for i in range(outer_folds)}

    for count, (train_index, test_index) in enumerate(outer_cv.split(X, y)):
        print(f"Iterazione {count}")

        # Creo modello nn e wrappo in sklearn
        model = KerasClassifier(build_fn = ff.create_sequential, input_size = (82, ), epochs = epochs, batch_size = 128, verbose = 0)

        # Inner cv
        inner_cv = StratifiedKFold(n_splits = inner_folds)

        # Seleziono i fold su cui lavorare
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Bilancio solamente il dataset di train (Forse da togliere?)
        if undersample: 
            X_train, y_train = init.undersample(X_train, y_train, ratio = ratio)

        # Faccio GridSearch su insieme di fold (Refit di base é a true, quindi best estimator é fittato su tutti e 4 i fold di train)
        # Refit permette di scegliere la metrica migliore su cui poi refittare il dataset
        grid = GridSearchCV(model, params, scoring = 'f1', cv = inner_cv, n_jobs = njobs).fit(X_train, y_train)

        # Modello migliore e relativi parametri
        best_model = grid.best_estimator_
        best_params = grid.best_params_
        best_result_idx = np.where(grid.cv_results_['rank_test_score'] == 1) 

        # Valuto prestazioni best model
        y_pred = best_model.predict(X_test)
        scores = classification_report(y_test, y_pred, output_dict = True)

        param_results[f'fold{count}'] = best_params
        best_models[f'fold{count}'] = best_model
        model_results[f'fold{count}']['f1'] = scores['1']['f1-score']
        model_results[f'fold{count}']['precision'] = scores['1']['recall']
        model_results[f'fold{count}']['recall'] = scores['1']['precision']
        model_results[f'fold{count}']['accuracy'] = scores['accuracy']

        print(f"f1-score test {scores['1']['f1-score']}, f1-score inner cv {grid.cv_results_['mean_test_score'][best_result_idx]} ({grid.cv_results_['std_test_score'][best_result_idx]}) con {best_params}")

    return best_models, model_results, param_results, 

def rnn_cross_validation(X, y, folds = 5, ratio = 0.2, h_size = 16, emb_size = 5, layers = 1):
    # Cross validation
    cv = StratifiedKFold(n_splits = folds)
    results = {'accuracy' : [], 'f1-score' : [], 'recall' : [], 'precision' : [], 'space' : []}

    # Dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(torch.cuda.get_device_name(0))

    for count, (train_index, test_index) in enumerate(cv.split(X, y), 1):
        # Inizializzo modello
        model = R.RNN(emb_size = emb_size, h_size = h_size, layers = layers).to(device)
        optimizer = torch.optim.Adamax(model.parameters())

        print("Indici Train: ", train_index, " Indici Test", test_index)
        print("Totale elementi: ", len(X))

        # Seleziono i fold su cui lavorare
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Bilancio solamente il dataset di train (Modifico anche quello originale?)
        X_train, y_train = init.undersample(X_train, y_train, ratio = ratio)

        # Training
        trainRNN.train(model, torch.tensor(X_train), torch.tensor(y_train), device = device, optimizer = optimizer)

        # Salvo pesi del modello
        # torch.save(model.state_dict(), f"rnn_pesi_modelli/{count}_RNN_emb{str(emb_size)}_hid{str(h_size)}")
        size = R.model_size(model, location = f"local_exec/classifier_testing/esperimenti_ffnn/risultati/pesi_modelli_addestrati/RNN_{count}.p")

        # Score del classificatore sul testing
        scores = R.score_report(model, X_test, y_test)

        # Aggiorno risultati delle metriche (Manca dev. st e size)
        results['accuracy'].append(scores['accuracy']) 
        results['space'].append(size)
        results['f1-score'].append(scores['1']['f1-score'])
        results['precision'].append(scores['1']['precision'])
        results['recall'].append(scores['1']['recall'])

    # Media sui 5 risultati
    mean_results = {key : sum(value) / folds for key, value in results.items()}
    std_results = {key : (sum([((x - mean_results[key]) ** 2) for x in value]) / folds) ** 0.5 for key, value in results.items()}

    return mean_results, std_results

def nn_cross_validation(X, y, folds, hidden_layer_size, learning_rate, ratio = 0.2, epochs = 30, callbacks = None, validation_split = None,  save_loss = True):
    # Cross validation
    cv = StratifiedKFold(n_splits = folds)
    results = {'accuracy' : [], 'f1-score' : [], 'recall' : [], 'precision' : [], 'space' : []}
    history = []

    for count, (train_index, test_index) in enumerate(cv.split(X, y), 1):
        # Inizializzo modello
        model = ff.create_sequential(input_size = (82, ), hidden_layer_size = hidden_layer_size, learning_rate = learning_rate, activation = 'relu')

        print("Indici Train: ", train_index, " Indici Test", test_index)
        print("Totale elementi: ", len(X))

        # Seleziono i fold su cui lavorare
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Bilancio solamente il dataset di train (Modifico anche quello originale?)
        X_train, y_train = init.undersample(X_train, y_train, ratio = ratio)

        # Training
        if validation_split is not None:
            X_train, y_train = helpers.shuffle(X_train, y_train)
        history.append(ff.train(model, X_train, y_train, cbs = callbacks, validation_split = validation_split, epochs = epochs))

        # Da modificare
        model = load_model('best_model.h5')

        # Valuto size modello (Metto location come param)
        size = ff.model_size(model, location = f"local_exec/classifier_testing/esperimenti_ffnn/risultati/pesi_modelli_addestrati/ff_{count}.p")
        # Score del classificatore sul testing
        scores = ff.evaluate(model, X_test, y_test, verbose = False)

        # Aggiorno risultati delle metriche (Manca dev. st e size)
        results['accuracy'].append(scores['accuracy']) 
        results['space'].append(size)
        results['f1-score'].append(scores['1']['f1-score'])
        results['precision'].append(scores['1']['precision'])
        results['recall'].append(scores['1']['recall'])

    # Salvo grafico loss
    if save_loss: ff.save_losses_plot(history[:3], f"test_plot_neuron{hidden_layer_size}_lr{learning_rate}", colors = ['r', 'b', 'g'])

    # Media sui 5 risultati
    mean_results = {key : sum(value) / folds for key, value in results.items()}
    std_results = {key : (sum([((x - mean_results[key]) ** 2) for x in value]) / folds) ** 0.5 for key, value in results.items()}

    return mean_results, std_results