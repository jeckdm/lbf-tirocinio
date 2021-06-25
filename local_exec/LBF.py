import time
import numpy as np
from BF import BloomFilter

def determine_tau(FPR_tau, probs_0):
  return np.percentile(np.array(probs_0),100*(1.-FPR_tau))

def build_LBF_classifier(probs0, probs1, phishing, FPR_tau):
    # probs0, probs1 = FFNN.get_classifier_probs(model, X, y)
    tau = determine_tau(FPR_tau, probs0)
    false_negs = []

    for i, url in enumerate(phishing):
        if(probs1[i] < tau): false_negs += [url]

    return false_negs, tau

def build_LBF_backup(false_negs, FPR, FPR_tau):
  num_false_negs = len(false_negs)
  FPR_B = (FPR-FPR_tau)/(1-FPR_tau)
  if(FPR_B <= 0):
    return "error"
  LBF_backup = BloomFilter(num_false_negs, FPR_B)
  for url in false_negs:
    LBF_backup.add(url)
    
  return LBF_backup

def test_LBF(LBF_backup, tau, testing_list, prediction):
    start = time.time()

    # Falsi positivi
    fps = 0             
    # Elementi totali
    total_elements = len(testing_list)

    for index, prob in enumerate(prediction):
        # Legit classificato come phishing
        if(prob > tau): result = True
        else: result = LBF_backup.check(prediction[index])

        if result: fps += 1

    end = time.time()
    # Tempo totale speso per accedere al BF di backup
    total_time = end - start

    # Ritorno risultati medi
    return fps / total_elements, LBF_backup.size / 8, total_time / total_elements
        
