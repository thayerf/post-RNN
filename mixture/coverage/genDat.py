### Learning how to calculate a parameter in a normal-normal model
import numpy as np


  
def gen_batch(m, n, sigma_theta):
  t = np.random.poisson(lam= 4.0, size = m)+1
  theta = [sigma_theta*np.random.randn(i) for i in t]
  z = [np.random.randint(i,size = n) for i in t]
  theta_true = np.zeros([m,n])
  for i in range(m):
        for j in range(n):
              theta_true[i,j]= theta[i][z[i][j]]
  X = theta_true + 0.1*np.random.randn(m,n)
  X = np.sort(X, axis = 1)
  labels = [np.max(theta[i]) for i in range(np.size(theta))]
  return({"X":X, "theta":labels})
  
  
  
  

