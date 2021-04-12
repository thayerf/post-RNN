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
## Build model
model = Sequential()
# Add input layer
model.add(SimpleRNN(nodes, return_sequences=True, input_shape=(None, 1)))
# Add additional hidden layers
for i in range(num_hidden-1):
      model.add(SimpleRNN(nodes, return_sequences=True))
# Add output layer
model.add(SimpleRNN(1, activation = "linear"))


## Define training data iterator
def genTraining(batch_size,train_n,sigma_theta):
    while True:
        # Get training data for step
        dat = gd.gen_batch(batch_size, train_n, sigma_theta)
        junk = ut.setup_nn_mat(dat)
        # We repeat the labels for each x in the sequence
        batch_labels = junk['outcome']
        batch_data = junk['W'].reshape((batch_size,train_n,1))
        yield batch_data,batch_labels

## Define callback for storing test miscoverage and model averaging
class average(callbacks.Callback):
  def __init__(self,test,exact,post,num_hidden, actual):
    self.test = test # test data
    self.exact = exact # exact quantiles
    self.miscover = [] # placeholder for miscoverage by epoch
    self.avg_miscover = [] # placeholder for miscoverage by epoch for average model
    self.post = post # posterior sd
    self.n = 0 # counter for burn-in
    self.avg_loss = []
    self.actual = actual
    # Build Averaging Model
    self.avg_model = Sequential()
    # Add input layer
    self.avg_model.add(SimpleRNN(32, return_sequences=True, input_shape=(None, 1)))
    # Add additional hidden layers
    for i in range(num_hidden-1):
          self.avg_model.add(SimpleRNN(32, return_sequences=True))
    # Add output layer
    self.avg_model.add(SimpleRNN(1))

  def on_epoch_end(self, epoch, logs={}):
    # Append epoch miscoverage and print out
    self.miscover.append(np.mean(np.abs(model.predict(self.test)-self.exact)/self.post))
    print("Test set miscover is {:01.6f}".format(self.miscover[-1]))
    # Get weights for model and average model
    new_weights = model.get_weights()
    curr_weights = self.avg_model.get_weights()
    # average appropriately if burn-in is done and print miscoverage
    if(self.n >burn_in):
          # We must loop over the elements to assign
          for i in range(np.size(curr_weights)):
                # This gives the mean of weights from epochs (burn_in) to (current)
                curr_weights[i] = ((1.0)/(self.n-burn_in))*new_weights[i] + ((self.n-1.0-burn_in)/(self.n -                  burn_in))*curr_weights[i]
          # Update the averaging weights
          self.avg_model.set_weights(curr_weights)
          # Append miscoverage to avg_miscover and print out, same with loss
          self.avg_miscover.append(np.mean(np.abs(self.avg_model.predict(self.test)-self.exact)/self.post))
          self.avg_loss.append(K.eval(pinball(self.avg_model.predict(self.test),self.actual)))
          print("Averaged model test set miscover is {:01.6f}".format(self.avg_miscover[-1]))
          print("Averaged model loss is {:01.6f}".format(self.avg_loss[-1]))
    # increment epoch number
    self.n = self.n+1
    
# Pinball loss for quantile tao
def pinball(y_true, y_pred):
    print(K.int_shape(y_pred)[1])
    pin = K.mean(K.maximum(y_true - y_pred, 0) * tao +
                 K.maximum(y_pred - y_true, 0) * (1 - tao))
    return pin
