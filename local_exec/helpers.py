import math
import numpy as np

# helper functions to do the math 

# given over-all FPR and  F_p, F_n of f, compute bits per element for BF
def solve_LBF_size(FPR, F_p, F_n, alpha=0.6185):
	ratio  = (FPR-F_p) / (1-F_p)
	b = F_n * math.log(ratio) / math.log(alpha)
	return b

def solve_SBF_size(FPR, F_p, F_n, alpha=0.6185):
	ratio = F_p / ((1-F_p)*(1/F_n-1))
	b_2 = F_n *  math.log(ratio) / math.log(alpha)
	constant = F_p + (1-F_p)* (alpha ** (b_2 / F_n))
	b_1 = math.log(FPR/constant) / math.log(alpha)
	return  b_1, b_2

def solve_FPR_LBF(b, F_p, F_n, alpha=0.6185):
	return  F_p+(1-F_p)* (alpha ** (b / F_n))

def solve_FPR_SBF(b1, b2, F_p, F_n, alpha=0.6185):
	return  (alpha ** b1)* (F_p + (1-F_p)* (alpha  ** (b2 / F_n)))

def determine_tau(FPR_tau, prob_list):
  return np.percentile(np.array(prob_list),100*(1.-FPR_tau))
  
# print(solve_LBF_size(0.005, 0.01, 0.5))
# print(solve_SBF_size(0.005, 0.01, 0.5))

# print(solve_FPR_SBF(2,6,0.01,0.5))