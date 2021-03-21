def BF_test_FPR(teFPR,phishing_URLs,testing_list,loc):   #dato un FPR teorico calcola e stampa l'FPR empirico calcolato usando filtri Bloom.
                                                      #risultato è un dizionario, se chiamato con save = true il risultato 
                                                      #  viene salvato come BF.npy in trained_NN/simulations.
  empFPR, BF_size, t = BF.run_BF(teFPR,phishing_URLs,testing_list)
  print("FPR", FPR, "size", BF_size, "time", t)
  BF = {"FPR": FPR, "size": BF_size, "time": t}
  if save==True:
    np.save(loc+"BF", BF)

def BF_test_size(phishing_URLs,testing_list):  #applica il filtro di bloom ad alcuni fpr pre-stabiliti e stampa size ottenuta
  BF_sizes = {}
# Aggiungo alcuni fpr
# Stampa grandezza del filtro in relazione al target fpr
  for fpr in fprs:
    BFo = BF.BloomFilter(len(phishing_URLs), fpr)
    BF_sizes[fpr] = BFo.size / 8
  print(BF_sizes)
