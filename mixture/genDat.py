### Learning how to calculate a parameter in a normal-normal model
import numpy as np
import scipy.special as sp


  
def gen_batch(m, n, sigma_theta):
  theta = sigma_theta * np.random.randn(m,5,1)
  z = np.random.randint(5,size = (m,n))
  theta_true = np.zeros([m,n])
  for i in range(m):
        for j in range(n):
              theta_true[i,j]= theta[i,z[i,j],0]
  X = theta_true + np.random.randn(m,n)
  labels = np.max(theta,axis=1)
  return({"X":X, "theta":labels})