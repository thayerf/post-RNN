import numpy as np
import genDat as gd
import utils as ut
# NETWORK STUFF

# Number of hidden layers
num_hidden = 8
# Number of nodes per hidden layer
nodes = 64
# Size of batches in each step
batch_size = 1000
# Training Sample Size
train_n = 250
# Testing Sample Size
test_n = 250
# Testing Batch Size
t_batch_size=2000
# Learning rate
step_size = 0.00001
# Learning rate decay
decay_size = .0001
# Number of steps per epoch
steps_per_epoch = 1
# Number of epochs
num_epochs = 2000
# Number of epochs to run before starting model averaging
burn_in = 100.0



# PRIOR PARAMETERS

# Specify mean and SD for level of log variance
b_k = 0
B_mu = 10
# Specify beta prior parameters for the persistance of the log variance
a_0 = 5
b_0 = 1.5
# Specify scalar on chi squared (1) prior for volatility of log variance
B_sigma = 1


# LOSS STUFF


# Pinball parameter
tao = 0.5
quant = np.array([0.5])



