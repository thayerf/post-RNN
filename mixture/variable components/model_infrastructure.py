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
inp = Input(shape=(None,1), name = "net_input")
l1= SimpleRNN(nodes, return_sequences=True)(inp)
l2= SimpleRNN(nodes, return_sequences=True)(l1)
l3= SimpleRNN(nodes, return_sequences=True)(l2)
l4= SimpleRNN(nodes, return_sequences=True)(l2)
# Add output layers
o1 = SimpleRNN(1, activation = "linear", name = "o1")(l3)
o2 = SimpleRNN(1, activation = "linear", name = "o2")(l4)
model = Model(inputs=[inp], outputs=[o1,o2])
# Define data iterator
def genTraining(batch_size,train_n,sigma_theta):
    while True:
        # Get training data for step
        dat = gd.gen_batch(batch_size, train_n, sigma_theta)
        junk = ut.setup_nn_mat(dat)
        # We repeat the labels for each x in the sequence
        batch_labels = junk['outcome']
        batch_data = junk['W'].reshape((batch_size,train_n,1))
        yield [batch_data,[batch_labels,batch_labels]]

def pinball_combined(y_true, y_pred):
    y_025, y_500,y_975 = tf.gather(y_pred,)
    pin = K.mean(K.maximum(y_true - y_500, 0) * 0.5 +
                 K.maximum(y_500 - y_true, 0) * 0.5)+K.mean(K.maximum(y_true - y_025, 0) * 0.025 + K.maximum(y_025 - y_true, 0) * 0.975)+K.mean(K.maximum(y_true - y_975, 0) * 0.975 + K.maximum(y_975 - y_true, 0) * 0.025)
                 
    return pin


def pin_025(y_true,y_pred):
      y_025 = y_pred
      pin = K.mean(K.maximum(.025 * (y_true-y_025), (.025-1) * (y_true-y_025)))
      return pin
def pin_975(y_true,y_pred):
      y_975 = y_pred
      pin = K.mean(K.maximum(.975 * (y_true-y_975), (.975-1) * (y_true-y_975)))
      return pin