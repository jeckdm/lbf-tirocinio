# Syspath
import sys

sys.path.append('.\local_exec')

# Librerie
import init
import trainRNN
import RNN as R
import pandas as pd
import argparse
import torch
import torch.nn as nn
from classifier_testing.esperimenti_ffnn_bf import analysis
from classifier_testing.esperimenti_ffnn_bf import helpers

def train(model, X_train, y_train, optimizer, criterion = nn.CrossEntropyLoss(), batch_size = 256):
  """ Train modificato per lavorare solamente su un modello, creata per non modifcare funzione originale """
  # Train and validate
  for epoch in range(30):
      _ = R.train(model, X_train, y_train, optimizer, criterion, batch_size)
      val_acc, val_loss = R.val(model, X_train, y_train, criterion, batch_size)
      if(epoch%10 == 0):
        print('[E{:4d}] Loss: {:.4f} | Acc: {:.4f}'.format(epoch, val_loss, val_acc))

  return model

def main(args): 
    # Parametri FFNN
    params = 16
    # Base sizes
    BF_sizes = {0.001 : 78616.625, 0.005 : 60299.75, 0.01 : 52411.0, 0.02 : 44522.375}
    # FPR test
    fprs = [0.001,0.005,0.01,0.02]
    fpr_ratios = [0.1*i for i in range(1,11)] # Ratio per LBF
    fpr_ratios2 = [1.*i for i in range(1,11)] # Ratio per SLBF
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Carico dataset
    X, y = init.load_data()
    
    # Codifica RNN
    X_rnn_enc, y_rnn_enc = init.RNN_encode(X, y, init.map_to_number(X))

    LBF_X_train, LBF_y_train, LBF_X_test, LBF_y_test, phishing_list, legit_testing_list = init.LBF_train_test_split(X, y, X_rnn_enc)

    # Creazione modelli
    rnn = R.RNN(emb_size = 5, h_size = params, layers = 1).to(device) 
    train(rnn, torch.tensor(LBF_X_train), torch.tensor(LBF_y_train), optimizer = torch.optim.Adamax(rnn.parameters()))
    model_size = R.model_size(rnn, f"local_exec/classifier_testing/risultati/models/RNN_{params}.pt", use_pickle = False)

    # Predizioni su test RNN
    probs1, probs0 = trainRNN.get_classifier_probs(rnn, torch.tensor(LBF_X_train), torch.tensor(LBF_y_train))
    y_hat, _ = rnn(torch.tensor(LBF_X_test).to(device))
    prediction = torch.sigmoid(y_hat[:,:,1])[:,149].squeeze().detach().cpu().numpy()

    # Analisi tau LBF
    false_negs, taus, fnrs = analysis.tau_analysis(probs0, probs1, fprs, fpr_ratios, phishing_list)
    # Trasformo fnrs in un df (Da inserire in qualche altra func)
    fnrs_df = pd.DataFrame(index=fpr_ratios, columns=fprs)
    for fpr in fprs:
      for fpr_ratio in fpr_ratios:
        fnrs_df.loc[fpr_ratio,fpr] = fnrs[(fpr, fpr_ratio)]

    helpers.create_graph(fnrs_df,f"RNN_{params}", path = f"local_exec/classifier_testing/risultati/plots/fnrs_LBF_RNN_{params}.png", xlabel = "Classifier FPR ratio "+r"$\epsilon_\tau/\epsilon$", ylabel = "Classifier FNR")

    helpers.LBF_analysis(fprs, fpr_ratios, false_negs, prediction, taus, legit_testing_list, model_size, BF_sizes, graph_name = f"RNN_{params}")

    # Analisi tau SLBF
    false_negs, taus, fnrs = analysis.tau_analysis(probs0, probs1, fprs, fpr_ratios + fpr_ratios2, phishing_list)
    # Trasformo fnrs in un df (Da inserire in qualche altra func)
    fnrs_df = pd.DataFrame(index=fpr_ratios + fpr_ratios2, columns=fprs)
    for fpr in fprs:
      for fpr_ratio in fpr_ratios + fpr_ratios2:
        fnrs_df.loc[fpr_ratio,fpr] = fnrs[(fpr, fpr_ratio)]

    helpers.create_graph(fnrs_df,f"RNN_{params}", path = f"local_exec/classifier_testing/risultati/plots/fnrs_SLBF_{params}.png", xlabel = "Classifier FPR ratio "+r"$\epsilon_\tau/\epsilon$", ylabel = "Classifier FNR")

    helpers.SLBF_analysis(fprs, fpr_ratios + fpr_ratios2, false_negs, prediction, taus, legit_testing_list, phishing_list, model_size, BF_sizes, graph_name = f"RNN_{params}")

if __name__ == "__main__":
    parse = argparse.ArgumentParser()
    parse.add_argument("--fprs", "-f", type = float, nargs = "+")
    parse.add_argument("--ratios", "-r", type = float, nargs = "+")
    parse.add_argument("--resultloc", "-l", type = str)

    args = parse.parse_args()

    main(args)