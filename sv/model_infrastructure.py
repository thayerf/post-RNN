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
from keras import initializers
# Build Model
model = Sequential()

# Add input layer
model.add(SimpleRNN(nodes, return_sequences=True, input_shape=(None, 1)))
# Add additional hidden layers
for i in range(num_hidden-1):
      model.add(SimpleRNN(nodes, return_sequences=True))
# Add output layer
model.add(SimpleRNN(1, activation = None))
# Define data iterator
def genTraining(batch_size,train_n,b_k,B_mu,a_0,b_0,B_sigma):
    while True:
        # Get training data for step
        dat = gd.gen_batch(batch_size, train_n, b_k,B_mu,a_0,b_0,B_sigma)
        junk = ut.setup_nn_mat(dat)
        # We repeat the labels for each x in the sequence
        if(np.linalg.norm(junk['outcome'])<np.inf and np.linalg.norm(junk['W']<np.inf)):
              batch_labels = junk['outcome']
              batch_data = junk['W'].reshape((batch_size,train_n,1))
        else:
              batch_labels = np.zeros([1,])
              batch_data = np.zeros([1,1,1])
        yield batch_data,batch_labels
        
        
        
        
        
        
        
# Define callback for model averaging
class average(callbacks.Callback):
  def __init__(self,test,exact,num_hidden):
    self.test = test   # get test data
    self.exact = exact # get exact labels
    self.avg_loss = []
    self.n = 0
    # Build Model
    self.avg_model = Sequential()
    # Add input layer
    self.avg_model.add(SimpleRNN(nodes, return_sequences=True, input_shape=(None, 1)))
    # Add additional hidden layers
    for i in range(num_hidden-1):
          self.avg_model.add(SimpleRNN(nodes, return_sequences=True))
    # Add output layer
    self.avg_model.add(SimpleRNN(1))

  def on_epoch_end(self, epoch, logs={}):
    new_weights = model.get_weights()
    curr_weights = self.avg_model.get_weights()
    if(self.n >burn_in):
          for i in range(np.size(curr_weights)):
                curr_weights[i] = (1.0)/(self.n-burn_in)*new_weights[i] + ((self.n-1.0-burn_in)/(self.n - burn_in))*curr_weights[i]
          self.avg_model.set_weights(curr_weights)
          self.avg_loss.append(K.eval(pinball(self.avg_model.predict(self.test)[:,0],self.exact)))
          print("Averaged model loss: {:01.6f}".format(self.avg_loss[-1]))
    
    self.n = self.n+1

def pinball(y_true, y_pred):
    pin = K.mean(K.maximum(y_true - y_pred, 0) * tao +
                 K.maximum(y_pred - y_true, 0) * (1 - tao))
    return pin