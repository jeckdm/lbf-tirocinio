import time
from BF import BloomFilter

def build_SLBF_initial(false_negs, FPR, FPR_tau, phishing_URLs):
    num_false_negs = len(false_negs)
    FPR_B0 = FPR/FPR_tau*(1.-num_false_negs/len(phishing_URLs))

    if(FPR_B0 <= 0 or FPR_B0 >= 1):
        return "error"
    SLBF_initial = BloomFilter(len(phishing_URLs), FPR_B0)
    for url in phishing_URLs:
        SLBF_initial.add(url)

    return SLBF_initial

def build_SLBF_backup(false_negs, FPR_tau, phishing_URLs):
    num_false_negs = len(false_negs)
    FPR_B = FPR_tau/((1-FPR_tau)*(len(phishing_URLs)/num_false_negs - 1))

    if(FPR_B <= 0):
        return "error"
    SLBF_backup = BloomFilter(num_false_negs, FPR_B)

    for url in false_negs:
        SLBF_backup.add(url)

    return SLBF_backup

def test_SLBF(SLBF_initial, SLBF_backup, testing_list, tau, prediction):
    fps = 0

    start = time.time()
    for ix, p in enumerate(prediction):
        result = SLBF_initial.check(testing_list[ix])

        if(result):
            if(p > tau): result = True
            else: result = SLBF_backup.check(testing_list[ix])

        if(result): fps += 1
    end = time.time()
    total_time = (end-start)

    avg_fp = fps / len(testing_list)
    
    # returns empirical FPR, BF size, and avg access time
    return avg_fp, SLBF_initial.size / 8,  SLBF_backup.size / 8, total_time