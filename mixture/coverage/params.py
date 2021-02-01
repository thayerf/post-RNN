import numpy as np
import genDat as gd
import utils as ut
# Set parameters

# Number of hidden layers
num_hidden = 4
# Number of nodes per hidden layer
nodes = 32
# Size of batches in each step
batch_size = 1000
# Training Sample Size
train_n = 250
# Testing Sample Size
test_n = 250
# Testing Batch Size
t_batch_size=500
# Learning rate
step_size = 0.001
# Learning rate decay
decay_size = .00000001
# Number of steps per epoch
steps_per_epoch = 1
# Number of epochs
num_epochs = 500
# Number of epochs to run before starting model averaging
burn_in = 100.0
# Prior Variance
sigma_theta = 0.5


# Pinball parameter
tao = 0.5
quant = np.array([0.5])



## Calulations, no need to edit
np.random.seed(12345)
sigma_posterior = pow(train_n + 1/pow(sigma_theta,2), -0.5)

# Create Testing Data
t_dat = gd.gen_batch(t_batch_size, test_n, sigma_theta)
junk = ut.setup_nn_mat(t_dat)
t_batch_labels = junk['outcome']


t_batch_data = junk['W'].reshape((t_batch_size,test_n,1))
t_sigma_posterior = pow(test_n + 1.0/pow(sigma_theta,2), -0.5)
t_exact_quants = pow(t_sigma_posterior,2)*(test_n*np.mean(t_batch_data,axis=1))
