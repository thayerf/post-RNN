0.### Learning how to calculate a parameter in a normal-normal model
import numpy as np


  
def gen_batch(m, n, b_k,B_mu,a_0,b_0,B_sigma):
  # Sample from the prior
  mu = B_mu*np.random.randn(m)
  phi = 2.0*np.random.beta(a_0,b_0,m)-1.0
  sigma = B_sigma*np.random.chisquare(1,m)
  # Create hidden volatility rv's and observed path rv's
  y = np.zeros([m,n])
  h = np.zeros([m,n])
  # Generate Initial Volatility
  h[:,0]= np.random.normal(mu, np.sqrt(sigma/(1.0-np.power(phi,2))))
  y[:,0]= np.random.normal(0,np.sqrt(np.exp(h[:,0])))
  # Sequentially generate values
  for i in range(n-1):
        h[:,i+1]= np.random.normal(mu+phi*(h[:,i]-mu),np.sqrt(sigma))
        y[:,i+1]= np.random.normal(0,np.sqrt(np.exp(h[:,i+1])))
  X = y
  labels = mu
  return({"X":X, "theta":labels})