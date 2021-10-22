import pandas as pd
import LBF
import SLBF

def tau_analysis(probs0, probs1, fprs, fprs_ratios, phishing, verbose = True):
    false_negs = {}
    taus = {}
    fnrs = {}

    for fpr in fprs:
        for fpr_ratio in fprs_ratios:
            false_negs[(fpr, fpr_ratio)], taus[(fpr, fpr_ratio)] = LBF.build_LBF_classifier(probs0, probs1, phishing, fpr*fpr_ratio)
            fnrs[(fpr, fpr_ratio)] = len(false_negs[(fpr, fpr_ratio)])/len(phishing)
            if verbose: print(f"FPR: {fpr}, FPR ratio: {fpr_ratio}, FNR: {len(false_negs[(fpr, fpr_ratio)])/len(phishing)}, Tau: {taus[(fpr, fpr_ratio)]}")

    return false_negs, taus, fnrs

def create_BFsbackup(fprs, fpr_ratios, false_negs):
    LBF_backups = {}

    # Per ognuno dei modelli salvo il filtro di backup costruito sulla base del fpr e fpr_ratio target
    for fpr in fprs:
        for fpr_ratio in fpr_ratios:
            LBF_backups[(fpr,fpr_ratio)] = LBF.build_LBF_backup(false_negs[(fpr,fpr_ratio)], fpr, fpr*fpr_ratio)
            if(LBF_backups[(fpr,fpr_ratio)] =='error'):
                continue

    return LBF_backups

def create_SLBF_filters(fprs, fpr_ratios, false_negs, phishing):
    SLBF_initials = {}
    SLBF_backups = {}

    for fpr in fprs:
        for fpr_ratio in fpr_ratios:
            c = (1. - len(false_negs[(fpr,fpr_ratio)]) / len(phishing))

            if(fpr_ratio < c or fpr * fpr_ratio > c):
                print(fpr_ratio, fpr, "bad fpr_tau")
                continue

            SLBF_initials[(fpr, fpr_ratio)] = SLBF.build_SLBF_initial(false_negs[(fpr, fpr_ratio)], fpr, fpr * fpr_ratio, phishing)
            SLBF_backups[(fpr, fpr_ratio)] = SLBF.build_SLBF_backup(false_negs[(fpr, fpr_ratio)], fpr * fpr_ratio, phishing)
    
    return SLBF_initials, SLBF_backups


def LBF_empirical_analysis(prediction, testing_list, fprs, fpr_ratios, taus, LBFs, verbose = True):
    true_fpr_LBF = pd.DataFrame(index = fpr_ratios, columns = fprs)
    sizes_LBF = pd.DataFrame(index = fpr_ratios, columns = fprs)
    times_LBF = pd.DataFrame(index = fpr_ratios, columns = fprs, dtype = float)

    for fpr in fprs:
        for fpr_ratio in fpr_ratios:
            try:
                # Calcolo risultati empirici
                BF_backup = LBFs[(fpr, fpr_ratio)]
                true_FPR, BF_backup_size, t = LBF.test_LBF(BF_backup, taus[(fpr, fpr_ratio)], testing_list, prediction)
                # Salvo risultati
                true_fpr_LBF.loc[fpr_ratio, fpr] = true_FPR
                sizes_LBF.loc[fpr_ratio, fpr] = BF_backup_size
                times_LBF.loc[fpr_ratio, fpr] = t

                if verbose:
                    print(f"FPR Target, FPR Ratio: ({fpr},{fpr_ratio}), FPR empirico: {true_FPR}, Size filtro backup: {BF_backup_size}, Tempo di accesso medio: {t}")
            except Exception as e:
                print(e)

    return true_fpr_LBF, sizes_LBF, times_LBF

def SLBF_empirical_analysis(predicition, testing_list, fprs, fpr_ratios, taus, SLBFs, verbose = True):
    true_fpr_SLBF = pd.DataFrame(index = fpr_ratios, columns = fprs)
    BF_initial_sizes_SLBF = pd.DataFrame(index = fpr_ratios, columns = fprs)
    BF_backup_sizes_SLBF = pd.DataFrame(index = fpr_ratios, columns = fprs)
    sizes_SLBF = pd.DataFrame(index = fpr_ratios, columns = fprs)
    times_SLBF = pd.DataFrame(index = fpr_ratios, columns = fprs, dtype = float)

    # Unpacking filtri SLBF
    SLBFs_initial, SLBFs_backup = SLBFs
    
    for fpr in fprs:
        for fpr_ratio in fpr_ratios:
            try:
                # Calcolo risultati empirici
                BF_initial = SLBFs_initial[(fpr, fpr_ratio)]
                BF_backup = SLBFs_backup[(fpr, fpr_ratio)]

                true_fpr, BF_initial_size, BF_backup_size, t = SLBF.test_SLBF(BF_initial, BF_backup, testing_list, taus[(fpr, fpr_ratio)], predicition)

                # Salvo risultati
                true_fpr_SLBF.loc[fpr_ratio, fpr] = true_fpr
                BF_initial_sizes_SLBF.loc[fpr_ratio, fpr] = BF_initial_size
                BF_backup_sizes_SLBF.loc[fpr_ratio, fpr] = BF_backup_size
                sizes_SLBF.loc[fpr_ratio, fpr] = BF_initial_size + BF_backup_size
                times_SLBF.loc[fpr_ratio, fpr] = t

                if verbose:
                    print(f"FPR Target, FPR Ratio: ({fpr},{fpr_ratio}), FPR empirico: {true_fpr}, Size filtro backup + initial: {BF_initial_size + BF_backup_size}, Tempo di accesso medio: {t}")

            except Exception as e:
                print(e)
    
    return true_fpr_SLBF, BF_initial_sizes_SLBF, BF_backup_sizes_SLBF, sizes_SLBF, times_SLBF



            