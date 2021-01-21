### Learning how to calculate a parameter in a normal-normal model
import numpy as np
import scipy.special as sp


## Generates an iid vector of observations x~N(theta,1)
## with theta ~ N(0,sigma^2)




def gen_batch(m, n, sigma_theta):
  theta = sigma_theta * np.random.randn(m,1)
  X = theta + np.random.randn(m,n)
  
  return({"X":X, "theta":theta})