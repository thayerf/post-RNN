import numpy as np
import genDat as gd
import utils as ut
# Set parameters

# Number of hidden layers
num_hidden = 4
# Number of nodes per hidden layer
nodes = 32
# Size of batches inS each step
batch_size = 100
# Training Sample Size
train_n = 500
# Testing Sample Size
test_n = 500
# Learning rate
step_size = 0.0001
# Number of steps per epoch
steps_per_epoch = 150
# Number of epochs
num_epochs = 200
# Number of epochs to run before starting model averaging
burn_in = 75.0
# Prior Variance
sigma_theta = 0.1


# Pinball parameter
tao = 0.5
quant = np.array([0.5])



## Calulations, no need to edit
np.random.seed(12345)
sigma_posterior = pow(train_n + 1/pow(sigma_theta,2), -0.5)

# Create Testing Data
t_dat = gd.gen_batch(5000, test_n, sigma_theta)
junk = ut.setup_nn_mat(t_dat)
t_batch_labels = junk['outcome']


t_batch_data = junk['W'].reshape((5000,test_n,1))
t_sigma_posterior = pow(test_n + 1.0/pow(sigma_theta,2), -0.5)
t_exact_quants = pow(t_sigma_posterior,2)*(test_n*np.mean(t_batch_data,axis=1))