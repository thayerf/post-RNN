### setting stuff up for the NN (not really used here)
import numpy as np
import scipy.stats as sps

def setup_nn_mat(dat):
    return({"outcome":dat['theta'], "W":dat['X']})

## Calculating exact conditional quantiles in the gaussian-gaussian scenario
def calc_quants(dat, quants, sigma_theta_squared):
    m,n = dat['X'].shape    
    n_quant = quants.size
    cond_quants = np.zeros((m,n_quant))
    for i in range(m):
        for j in range(n_quant):
            x_bar = np.mean(dat['X'][i,:])
            mean_ppf = x_bar*sigma_theta_squared/(pow(n,-1) + sigma_theta_squared)
            sd_ppf = pow(n + 1/sigma_theta_squared, -0.5)
            cond_quants[i,j] = sps.norm.ppf(quants[j], mean_ppf, sd_ppf)
    return(cond_quants)

## evaluating the difference between predicted and true quantiles
def eval_diff(pred_quants, exact_quants):
   diff_abs = np.mean(np.abs(pred_quants - exact_quants))
   return(diff_abs)
