import matplotlib.pyplot as plt
from cycler import cycler
from classifier_testing.esperimenti_ffnn_bf import analysis

def create_graph(params, title, xlabel, ylabel, path):
    plt.figure(figsize=(8, 6))

    for fpr in params:
        plt.plot(params[fpr], label = f"FPR: {fpr}")

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(loc='best')

    plt.savefig(path)

def size_graph(params, base_sizes, title, xlabel, ylabel, path):
    plt.figure(figsize=(8, 6))

    for fpr in params:
        line, = plt.plot(params[fpr], label = f"FPR: {fpr}", zorder = 20)
        plt.axhline(y = base_sizes[fpr], xmin = -10, xmax = 20, color = line.get_color(), alpha = 0.5, linestyle = '--', zorder = 0) # Molto brutto, solo per provare
        
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(loc='best')

    plt.savefig(path)

def create_SLBF_size_graph(tot_sizes, BF_initial_sizes, BF_backup_sizes, title, xlabel, ylabel, path):
    plt.figure(figsize=(8, 6))

    for fpr in tot_sizes:
        line, = plt.plot(BF_initial_sizes[fpr], label = f"Initial - FPR: {fpr}", linestyle = '--', marker = 'o', zorder = 10)
        plt.plot(BF_backup_sizes[fpr],  label = f"Backup - FPR: {fpr}", linestyle = ':', marker = 's', color = line.get_color(), zorder = 10)

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.legend(loc='best')

    plt.savefig(path)

def create_comparison_size_graph(true_fpr_LBFs, true_fpr_SLBFs, sizes_LBFs, sizes_SLBFs, path):
    x_LBF, y_LBF, z_LBF = [], [], []
    x_SLBF, y_SLBF, z_SLBF = [], [], []

    plt.figure(figsize=(8, 6))

    for j1 in range(true_fpr_LBFs.shape[1]):
        for j0 in range(true_fpr_LBFs.shape[0]):
            x_LBF += [true_fpr_LBFs.iloc[j0,j1]]
            y_LBF += [sizes_LBFs.iloc[j0,j1]]
        z_LBF += [min(sizes_LBFs.iloc[:,j1])]
    
    for j1 in range(true_fpr_SLBFs.shape[1]):
        for j0 in range(true_fpr_SLBFs.shape[0]):
            if(true_fpr_SLBFs.index[j0] >= 1.):
                x_SLBF += [true_fpr_SLBFs.iloc[j0,j1]]
                y_SLBF += [sizes_SLBFs.iloc[j0,j1]]
        z_SLBF += [min(sizes_SLBFs.dropna().iloc[:,j1])]

    plt.scatter(x_LBF,y_LBF, label = 'LBF', marker='.')
    plt.scatter(x_SLBF,y_SLBF, label = 'SLBF', marker='.')
    plt.plot(true_fpr_LBFs.columns,z_LBF)
    plt.plot(true_fpr_SLBFs.columns,z_SLBF)
    plt.title("Size comparison")
    plt.xlabel("Observed FPR")
    plt.ylabel("Total Size (bytes)")
    plt.xlim(left=0, right=0.03)

    plt.savefig(path)

def LBF_analysis(fprs, fpr_ratios, false_negs, prediction, taus, legit_testing_list, model_size, BF_sizes, graph_name = None):
    # Creazione BF di backup
    BF_backups = analysis.create_BFsbackup(fprs, fpr_ratios, false_negs)
    # Analisi empirica di Classificatore + BF
    true_fpr_LBF, sizes_LBF, times_LBF = analysis.LBF_empirical_analysis(prediction, legit_testing_list, fprs, fpr_ratios, taus, BF_backups)
    sizes_LBF += model_size 

    if graph_name is not None:
        create_graph(true_fpr_LBF, f"{graph_name} ", path = f"local_exec/classifier_testing/risultati/plots/true_fpr_LBF_{graph_name}.png", xlabel = "Classifier FPR ratio "+r"$\epsilon_\tau/\epsilon$", ylabel = "Overall FPR")
        size_graph(sizes_LBF, BF_sizes, f"{graph_name}", path = f"local_exec/classifier_testing/risultati/plots/size_LBF_{graph_name}.png", xlabel = "Classifier FPR ratio "+r"$\epsilon_\tau/\epsilon$", ylabel = "Total size")
    
    return true_fpr_LBF, sizes_LBF

def SLBF_analysis(fprs, fpr_ratios, false_negs, prediction, taus, legit_testing_list, phishing_list, model_size, BF_sizes, graph_name = None):
    # Creazione filtro iniziale e di backup
    SLBF_filters = analysis.create_SLBF_filters(fprs, fpr_ratios, false_negs, phishing_list)
    # Analisi empirica BF iniziale + Classifcatore + BF finale
    true_fpr_SLBF, BF_initial_sizes_SLBF, BF_backup_sizes_SLBF, sizes_SLBF, times_SLBF = analysis.SLBF_empirical_analysis(prediction, legit_testing_list, fprs, fpr_ratios, taus, SLBF_filters)
    sizes_SLBF += model_size

    if graph_name is not None:
        create_graph(true_fpr_SLBF, f"{graph_name} ", path = f"local_exec/classifier_testing/risultati/plots/true_fpr_SLBF_{graph_name}.png", xlabel = "Classifier FPR ratio "+r"$\epsilon_\tau/\epsilon$", ylabel = "Overall FPR")
        size_graph(sizes_SLBF, BF_sizes, f"{graph_name}", path = f"local_exec/classifier_testing/risultati/plots/size_SLBF_{graph_name}.png", xlabel = "Classifier FPR ratio "+r"$\epsilon_\tau/\epsilon$", ylabel = "Total size")
        create_SLBF_size_graph(sizes_SLBF, BF_initial_sizes_SLBF, BF_backup_sizes_SLBF, title = f"{graph_name}", path = f"local_exec/classifier_testing/risultati/plots/size_comparison_SLBF_{graph_name}.png", xlabel = "Classifier FPR ratio "+r"$\epsilon_\tau/\epsilon$", ylabel = "Total size")
    
    return true_fpr_SLBF, sizes_SLBF