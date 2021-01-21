### Learning how to calculate a parameter in a normal-normal model
import numpy as np


  
def gen_batch(m):
      
  # Sample from the prior
  beta = np.random.gamma(9,.05, size = m)
  gamma = np.random.gamma(3,.05, size = m)
  # Initialize population
  N = 100
  I = np.zeros([m,10001])
  S = np.zeros([m,10001])
  I[:,0] = 2
  S[:,0] = 98
  dt = 0.01
  for i in range(10000):
        w_1 = np.random.normal(size= m)
        w_2 = np.random.normal(size= m)
        S[:,i+1] = np.maximum((S[:,i] - dt * beta*S[:,i]*I[:,i]/N - np.sqrt(dt * beta*S[:,i]*I[:,i]/N)* w_1),0)
        I[:,i+1] = np.maximum((I[:,i] + dt * beta*S[:,i]*I[:,i]/N - dt * gamma * I[:,i] + np.sqrt(dt)*(np.sqrt(beta*S[:,i]*I[:,i]/N)*w_1- np.sqrt(gamma* I[:,i])*w_2)),0)
  # #####
  labels = beta/gamma
  obs_I = I[:,0:10000:100]
  obs_S = S[:,0:10000:100]
  X = np.zeros([m,100,2])
  X[:,:,0] = obs_I
  X[:,:,1] = obs_S
  
  return({"X":X, "theta":labels})
  