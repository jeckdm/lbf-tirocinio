import matplotlib.pyplot as plt
import numpy as np
import analysis

def shuffle(X, y, seed = 0):
    """ Ritorna una permutazione di X,y generata con il seed in input"""

    np.random.seed(seed)
    perm = np.random.permutation(len(X))

    X = X[perm]
    y = y[perm]

    return X, y

def custom_plot(datas, base_sizes = None, ax = None, **kwargs):
    ax = ax or plt.gca()

    for d in datas:
        line, = ax.plot(datas[d], **kwargs)
        if base_sizes is not None : ax.axhline(y = base_sizes[d], color = line.get_color(), alpha = 0.5, linestyle = '--')
    
    return ax

def custom_SLBF_plot(BF_initial_sizes, BF_backup_sizes, ax = None, kwargs_init = {}, kwargs_backup = {}):
    ax = ax or plt.gca()

    for fpr in BF_initial_sizes:
        line, = ax.plot(BF_initial_sizes[fpr], **kwargs_init)
        ax.plot(BF_backup_sizes[fpr], color = line.get_color(), **kwargs_backup)

    return ax

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

def LBF_analysis(fprs, fpr_ratios, false_negs, prediction, taus, legit_testing_list, model_size):
    # Creazione BF di backup
    BF_backups = analysis.create_BFsbackup(fprs, fpr_ratios, false_negs)
    # Analisi empirica di Classificatore + BF
    true_fpr_LBF, sizes_LBF, times_LBF = analysis.LBF_empirical_analysis(prediction, legit_testing_list, fprs, fpr_ratios, taus, BF_backups)
    sizes_LBF += model_size 

    return true_fpr_LBF, sizes_LBF, times_LBF

def SLBF_analysis(fprs, fpr_ratios, false_negs, prediction, taus, legit_testing_list, phishing_list, model_size):
    # Creazione filtro iniziale e di backup
    SLBF_filters = analysis.create_SLBF_filters(fprs, fpr_ratios, false_negs, phishing_list)
    # Analisi empirica BF iniziale + Classifcatore + BF finale
    true_fpr_SLBF, BF_initial_sizes_SLBF, BF_backup_sizes_SLBF, sizes_SLBF, times_SLBF = analysis.SLBF_empirical_analysis(prediction, legit_testing_list, fprs, fpr_ratios, taus, SLBF_filters)
    sizes_SLBF += model_size
    
    return true_fpr_SLBF, sizes_SLBF, BF_initial_sizes_SLBF, BF_backup_sizes_SLBF, times_SLBF

def generate_LBF_graphs(name, false_negs, true_FPR, base_sizes, sizes, save_loc = None):
    fig, axes = plt.subplots(3, figsize = (5,12))

    custom_plot(ax = axes[0], datas = false_negs)
    custom_plot(ax = axes[1], datas = true_FPR)
    custom_plot(ax = axes[2], datas = sizes, base_sizes = base_sizes)

    axes[0].set_title(label = f'{name} FNR Ratio')
    axes[1].set_title(label = f'{name} True FPR')
    axes[2].set_title(label = f'{name} Sizes')

    for ax in axes: 
        ax.set_xlim(0.0, 1.1)

    plt.tight_layout()

    if save_loc is not None: 
        plt.savefig(save_loc)

def generate_SLBF_graphs(name, false_negs, true_FPR, base_sizes, sizes, initial_sizes, backup_sizes, save_loc = None):
    fig, axes = plt.subplots(4, figsize = (5,12))

    custom_plot(ax = axes[0], datas = false_negs)
    custom_plot(ax = axes[1], datas = true_FPR)
    custom_plot(ax = axes[2], datas = sizes, base_sizes = base_sizes)
    custom_SLBF_plot(ax = axes[3], BF_initial_sizes = initial_sizes, BF_backup_sizes = backup_sizes, kwargs_init = {'linestyle' : '--', 'marker' : 'o'}, kwargs_backup = {'linestyle' : ':', 'marker' : 's'})

    axes[0].set_title(label = f'{name} FNR Ratio')
    axes[1].set_title(label = f'{name} True FPR')
    axes[2].set_title(label = f'{name} Sizes')
    axes[3].set_title(label = f'{name} Filters comparison')

    for ax in axes: 
        ax.set_xlim(0.0, 10.5)

    plt.tight_layout()

    if save_loc is not None: 
        plt.savefig(save_loc)