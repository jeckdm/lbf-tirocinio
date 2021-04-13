import matplotlib.pyplot as plt
import seaborn as sns
import trainRNN
import config

'''
Genera grafici rappresentate la distribuzione dei valori degli score su X_train dei classificatori in input.
'''
def classifier_graph(models, X_train, y_train, name):
  f, ax = plt.subplots(1, len(models), figsize=(12,3))

  for i, key in enumerate(models):
    probs = trainRNN.get_classifier_probs(models[key], X_train, y_train)
    sns.histplot(probs, ax = ax[i])

  f.savefig(config.loc_plots+name)

'''
Genera grafici su LBF per ognuno dei (fpr, fprs_ratio)
Nello specifico genera i seguenti grafici:
- Andamento FN rate al variare del FPR ratio per ognuno degli FPR target
- Overall FPR empirico al variare del FPR ratio per ognuno degli FPR target
- Dimensione totale del SLBF al variare del FPR ratio per ognuno degli FPR target
I grafici vengono salvati in plot_loc
'''
def LBF_graph(fnrs, true_fpr_LBF, sizes_LBF,sizes_struct_LBF, name, falseN=True, FPR=True, Size=True,size_struct=True):
  if(falseN):
    graph(fnrs, "Classifier False Negative Rate", name + "_classifier_FNR.png", )
  if(FPR):
    graph(true_fpr_LBF, "Overall FPR", name + "_fpr.png", )
  if(Size):
    graph(sizes_LBF, "Total Size of LBF",  name + "_size.png")
  if(size_struct):
    graph(sizes_struct_LBF, "Sizes elements LBF", name + "_size_struct.png")

def graph(params, title, path):
  f,ax = plt.subplots(1, len(config.h_sizes), figsize=(12,3))
  for i in range(len(config.h_sizes)):
    params[i].plot(ax=ax[i])
    ax[i].set_xlabel("Classifier FPR ratio "+r"$\epsilon_\tau/\epsilon$")
    ax[i].set_ylabel(title)
    ax[i].set_title("LBF with "+str(config.h_sizes[i])+" dimensional GRU")
    ax[i].legend(fontsize='xx-small')
  plt.tight_layout()
  f.savefig(config.loc_plots+path)