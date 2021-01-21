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
inputs = Input((None, 1))
x = SimpleRNN(32, return_sequences=True)(inputs)
# Add additional hidden layers
for i in range(num_hidden - 1):
    x = SimpleRNN(nodes, return_sequences=True)(x)
# Add output layer
o1 = SimpleRNN(1, activation="linear")(x)
o2 = SimpleRNN(1, activation="linear")(x)
o3 = SimpleRNN(1, activation="linear")(x)
o4 = SimpleRNN(1, activation="linear")(x)
o5 = SimpleRNN(1, activation="linear")(x)
o6 = SimpleRNN(1, activation="linear")(x)
o7 = SimpleRNN(1, activation="linear")(x)
o8 = SimpleRNN(1, activation="linear")(x)
o9 = SimpleRNN(1, activation="linear")(x)
model = Model(inputs, [o1, o2, o3, o4, o5, o6, o7, o8, o9])


## Define callback for storing test miscoverage and model averaging
class average(callbacks.Callback):
    def __init__(self, test, exact, post, num_hidden, actual):
        self.test = test  # test data
        self.exact = exact  # exact quantiles
        self.miscover = []  # placeholder for miscoverage by epoch
        self.avg_miscover = []  # placeholder for miscoverage by epoch for average model
        self.post = post  # posterior sd
        self.n = 0  # counter for burn-in
        self.avg_loss = []
        self.actual = actual
        # Build Averaging Model
        self.avg_model = Sequential()
        # Add input layer
        self.avg_model.add(SimpleRNN(32, return_sequences=True, input_shape=(None, 1)))
        # Add additional hidden layers
        for i in range(num_hidden - 1):
            self.avg_model.add(SimpleRNN(32, return_sequences=True))
        # Add output layer
        self.avg_model.add(SimpleRNN(1))

    def on_epoch_end(self, epoch, logs={}):
        # Append epoch miscoverage and print out
        self.miscover.append(
            np.mean(np.abs(model.predict(self.test) - self.exact) / self.post)
        )
        print("Test set miscover is {:01.6f}".format(self.miscover[-1]))
        # Get weights for model and average model
        new_weights = model.get_weights()
        curr_weights = self.avg_model.get_weights()
        # average appropriately if burn-in is done and print miscoverage
        if self.n > burn_in:
            # We must loop over the elements to assign
            for i in range(np.size(curr_weights)):
                # This gives the mean of weights from epochs (burn_in) to (current)
                curr_weights[i] = ((1.0) / (self.n - burn_in)) * new_weights[i] + (
                    (self.n - 1.0 - burn_in) / (self.n - burn_in)
                ) * curr_weights[i]
            # Update the averaging weights
            self.avg_model.set_weights(curr_weights)
            # Append miscoverage to avg_miscover and print out, same with loss
            self.avg_miscover.append(
                np.mean(
                    np.abs(self.avg_model.predict(self.test) - self.exact) / self.post
                )
            )
            self.avg_loss.append(
                K.eval(pinball(self.avg_model.predict(self.test), self.actual))
            )
            print(
                "Averaged model test set miscover is {:01.6f}".format(
                    self.avg_miscover[-1]
                )
            )
            print("Averaged model loss is {:01.6f}".format(self.avg_loss[-1]))
        # increment epoch number
        self.n = self.n + 1


# Pinball loss for quantile tao
def pinball(tao, y_true, y_pred, sample_weight=None):
    pin = K.mean(
        K.maximum(y_true - y_pred, 0) * tao + K.maximum(y_pred - y_true, 0) * (1 - tao)
    )
    return pin
