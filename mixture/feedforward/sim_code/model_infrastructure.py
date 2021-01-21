from keras import backend as K
from keras import callbacks
import utils as ut
import genDat as gd
import numpy as np
from params import *
from keras.models import Sequential, Model
from keras.layers import LSTM, Dense, TimeDistributed, SimpleRNN, Input
from keras.utils import to_categorical
from keras import optimizers
import tensorflow as tf
# Build Model


# Add input layer
## Build model
model = Sequential()
# Add input layer
model.add(Dense(nodes, input_dim=250, init = 'uniform', activation = 'relu'))
# Add additional hidden layers
for i in range(num_hidden-1):
      model.add(Dense(nodes, activation = 'relu', kernel_initializer = 'uniform'))
# Add output layer
model.add(Dense(1, activation = "linear"))
# Define data iterator
def genTraining(batch_size,train_n,sigma_theta):
    while True:
        # Get training data for step
        dat = gd.gen_batch(batch_size, train_n, sigma_theta)
        junk = ut.setup_nn_mat(dat)
        # We repeat the labels for each x in the sequence
        batch_labels = junk['outcome']
        batch_data = junk['W'].reshape((batch_size,train_n))
        yield [batch_data,[batch_labels,batch_labels]]

def pinball_combined(y_true, y_pred):
    y_025, y_500,y_975 = tf.gather(y_pred,)
    pin = K.mean(K.maximum(y_true - y_500, 0) * 0.5 +
                 K.maximum(y_500 - y_true, 0) * 0.5)+K.mean(K.maximum(y_true - y_025, 0) * 0.025 + K.maximum(y_025 - y_true, 0) * 0.975)+K.mean(K.maximum(y_true - y_975, 0) * 0.975 + K.maximum(y_975 - y_true, 0) * 0.025)
                 
    return pin


def pin_5(y_true,y_pred):
      y_025 = y_pred
      pin = K.mean(K.maximum(.5 * (y_true-y_025), (-0.5) * (y_true-y_025)))
      return pin