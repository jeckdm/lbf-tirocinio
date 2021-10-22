import matplotlib.pyplot as plt
import numpy as np
import analysis
import pandas as pd

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
        line, = ax.plot(datas[d], label = f'{d}', **kwargs)
        if base_sizes is not None : ax.axhline(y = base_sizes[d], color = line.get_color(), alpha = 0.5, linestyle = '--')
    
    return ax

def custom_SLBF_plot(BF_initial_sizes, BF_backup_sizes, ax = None, kwargs_init = {}, kwargs_backup = {}):
    ax = ax or plt.gca()

    for fpr in BF_initial_sizes:
        line, = ax.plot(BF_initial_sizes[fpr], label = f'{fpr} Initial', **kwargs_init)
        ax.plot(BF_backup_sizes[fpr], label = f'{fpr} Backup', color = line.get_color(), **kwargs_backup)

    return ax


def LBF_analysis(fprs, fpr_ratios, fnrs, false_negs, prediction, taus, legit_testing_list, model_size):
    # Risultati
    df_best_result = {fpr : {'size' : 0, 'model-size' : 0, 'time' : 0} for fpr in fprs}

    fnrs_df = pd.DataFrame(index=fpr_ratios, columns=fprs)
    for fpr in fprs:
        for fpr_ratio in fpr_ratios:
            fnrs_df.loc[fpr_ratio,fpr] = fnrs[(fpr, fpr_ratio)]

    # Creazione BF di backup
    BF_backups = analysis.create_BFsbackup(fprs, fpr_ratios, false_negs)
    # Analisi empirica di Classificatore + BF
    true_fpr_LBF, sizes_LBF, times_LBF = analysis.LBF_empirical_analysis(prediction, legit_testing_list, fprs, fpr_ratios, taus, BF_backups)
    sizes_LBF += model_size 

    for fpr in fprs :
        best_fpr_ratio = pd.to_numeric(sizes_LBF[fpr].dropna(), downcast = 'float').idxmin()

        best_size = sizes_LBF.loc[best_fpr_ratio, fpr]
        best_time = times_LBF.loc[best_fpr_ratio, fpr]

        df_best_result[fpr]['size'] = best_size / 1024
        df_best_result[fpr]['model-size'] = model_size / 1024
        df_best_result[fpr]['time'] = best_time

    return fnrs_df, true_fpr_LBF, sizes_LBF, times_LBF, pd.DataFrame.from_dict(df_best_result)

def SLBF_analysis(fprs, fpr_ratios, fnrs, false_negs, prediction, taus, legit_testing_list, phishing_list, model_size):
    # Risultati
    df_result = {fpr : {'size' : 0, 'initial-size' : 0, 'backup-size' : 0, 'model-size' : 0, 'time' : 0} for fpr in fprs}

    fnrs_df = pd.DataFrame(index=fpr_ratios, columns=fprs)
    for fpr in fprs:
        for fpr_ratio in fpr_ratios:
            fnrs_df.loc[fpr_ratio,fpr] = fnrs[(fpr, fpr_ratio)]

    # Creazione filtro iniziale e di backup
    SLBF_filters = analysis.create_SLBF_filters(fprs, fpr_ratios, false_negs, phishing_list)
    # Analisi empirica BF iniziale + Classifcatore + BF finale
    true_fpr_SLBF, BF_initial_sizes_SLBF, BF_backup_sizes_SLBF, sizes_SLBF, times_SLBF = analysis.SLBF_empirical_analysis(prediction, legit_testing_list, fprs, fpr_ratios, taus, SLBF_filters)
    sizes_SLBF += model_size
    
    for fpr in fprs:
        best_fpr_ratio = pd.to_numeric(sizes_SLBF[fpr].dropna(), downcast = 'float').idxmin()

        best_size = sizes_SLBF.loc[best_fpr_ratio, fpr] / 1024
        best_isize = BF_initial_sizes_SLBF.loc[best_fpr_ratio, fpr] / 1024
        best_bsize = BF_backup_sizes_SLBF.loc[best_fpr_ratio, fpr] / 1024
        best_time =  times_SLBF.loc[best_fpr_ratio, fpr]

        print(best_fpr_ratio, best_size, best_isize, best_bsize, best_time)
        df_result[fpr]['size'] = best_size
        df_result[fpr]['model-size'] = model_size / 1024
        df_result[fpr]['initial-size'] = best_isize
        df_result[fpr]['backup-size'] = best_bsize
        df_result[fpr]['time'] = best_time

    return fnrs_df, true_fpr_SLBF, BF_initial_sizes_SLBF, BF_backup_sizes_SLBF, sizes_SLBF, times_SLBF, pd.DataFrame.from_dict(df_result)


def generate_LBF_graphs(name, false_negs, true_FPR, base_sizes, sizes, params, save_loc = None):
    fig, axes = plt.subplots(3, figsize = (5,10), sharex = True)

    custom_plot(ax = axes[0], datas = false_negs)
    custom_plot(ax = axes[1], datas = true_FPR)
    custom_plot(ax = axes[2], datas = sizes, base_sizes = base_sizes)

    for idx, ax in enumerate(axes):
        ax.set_title(label = f'{name} {params["title"][idx]}')
        ax.set_ylim(params["ylim"][idx])
        ax.set_ylabel(params["ylabel"][idx])
        ax.legend(fontsize = params["legend"][idx])

    for ax in axes: 
        ax.set_xlabel(r"Classifier FPR ratio $\frac{\epsilon_{\tau}}{\epsilon}$")

    plt.tight_layout()

    if save_loc is not None: 
        plt.savefig(save_loc)

def generate_SLBF_graphs(name, false_negs, true_FPR, base_sizes, sizes, initial_sizes, backup_sizes, params, save_loc = None):
    fig, axes = plt.subplots(4, figsize = (5,13), sharex = True)

    custom_plot(ax = axes[0], datas = false_negs)
    custom_plot(ax = axes[1], datas = true_FPR)
    custom_plot(ax = axes[2], datas = sizes, base_sizes = base_sizes)
    custom_SLBF_plot(ax = axes[3], BF_initial_sizes = initial_sizes, BF_backup_sizes = backup_sizes, kwargs_init = {'linestyle' : '--', 'marker' : 'o'}, kwargs_backup = {'linestyle' : ':', 'marker' : 's'})

    for idx, ax in enumerate(axes):
        ax.set_title(label = f'{name} {params["title"][idx]}')
        ax.set_ylim(params["ylim"][idx])
        ax.set_ylabel(params["ylabel"][idx])
        ax.legend(fontsize = params["legend"][idx]['fontsize'], ncol = params["legend"][idx]["ncol"])

    for ax in axes: 
        ax.set_xlabel(r"Classifier FPR ratio $\frac{\epsilon_{\tau}}{\epsilon}$")

    plt.tight_layout()

    if save_loc is not None: 
        plt.savefig(save_loc)

def generate_comparison_graph(sizes, fprs, save_loc = None):
    '''
    {
        'FFNN20' : {0.001 : ... , 0.05 : ...}
        'RNN16'  : {0.001 : ... , 0.05 : ...}
        'FFNN30' : {0.001 : ... , 0.05 : ...}
        ...
    }
    '''
    fig, axes = plt.subplots(1, figsize = (5,10))

    data = pd.DataFrame(index = fprs, columns = sizes.keys())

    for key, value in sizes.items():
        for fpr in fprs:
            data[fpr, key] = min(value[fpr])
    
    custom_plot(ax = axes[0], datas = data)
