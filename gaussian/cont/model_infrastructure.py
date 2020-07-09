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

## Build model
# Add input layer
inputs = Input((None, 2))
t = Input(shape=(1,))
x = SimpleRNN(32, return_sequences=True)(inputs)
# Add additional hidden layers
for i in range(num_hidden - 1):
    x = SimpleRNN(nodes, return_sequences=True)(x)
# Add output layer
o1 = SimpleRNN(1, activation="linear")(x)
model = Model([inputs,t], o1)

# Pinball loss for quantile tao
def pinball(tao, y_true, y_pred, sample_weight=None):
    pin = K.mean(
        K.maximum(y_true - y_pred, 0) * tao + K.maximum(y_pred - y_true, 0) * (1 - tao)
    )
    return pin
