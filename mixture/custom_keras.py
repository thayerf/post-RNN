from keras import backend as K
from keras import callbacks
import utils as ut
import genDat as gd
import numpy as np
from params import *
from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, SimpleRNN, Input
from keras.utils import to_categorical
from keras import optimizers
# Define callbacks and generators
# Define training data iterator
model = Sequential()

def genTraining(batch_size,train_n,sigma_theta):
    while True:
        # Get training data for step
        dat = gd.gen_batch(batch_size, train_n, sigma_theta)
        junk = ut.setup_nn_mat(dat)
        # We repeat the labels for each x in the sequence
        batch_labels = junk['outcome']
        batch_data = junk['W'].reshape((batch_size,train_n,1))
        yield batch_data,batch_labels
# Define callback for storing test miscoverage
class miscover(callbacks.Callback):
  def __init__(self, test,exact):
    self.test = test   # get test data
    self.exact = exact
    self.miscover = []

  def on_epoch_end(self, epoch, logs={}):
    self.miscover.append(np.mean(np.abs(model.predict(self.test)>self.exact)))
    print("Normal model estimate is greater than label value {:01.6f} of the time".format(self.miscover[-1]))
    
class average(callbacks.Callback):
  def __init__(self,test,exact,num_hidden):
    self.test = test   # get test data
    self.exact = exact
    self.miscover = []
    self.n = 0
    # Build Model
    self.avg_model = Sequential()
    # Add input layer
    self.avg_model.add(SimpleRNN(32, return_sequences=True, input_shape=(None, 1)))
    # Add additional hidden layers
    for i in range(num_hidden-1):
          self.avg_model.add(SimpleRNN(nodes, return_sequences=True))
    self.avg_model.add(SimpleRNN(1))

  def on_epoch_end(self, epoch, logs={}):
    new_weights = model.get_weights()
    curr_weights = self.avg_model.get_weights()
    if(self.n >burn_in):
          for i in range(np.size(curr_weights)):
                curr_weights[i] = (1.0)/(self.n-burn_in)*new_weights[i] + ((self.n-1.0-burn_in)/(self.n - burn_in))*curr_weights[i]
          self.avg_model.set_weights(curr_weights)
          self.miscover.append(np.mean((self.avg_model.predict(self.test)>self.exact)))
          print("Averaged model estimate is greater than label value {:01.6f} of the time".format(self.miscover[-1]))
          print("Averaged model loss: {:01.6f}".format(np.mean(np.abs(self.avg_model.predict(self.test)-self.exact))))
    
    self.n = self.n+1

def pinball(y_true, y_pred):
    pin = K.mean(K.maximum(y_true - y_pred, 0) * tao +
                 K.maximum(y_pred - y_true, 0) * (1 - tao))
    return pin