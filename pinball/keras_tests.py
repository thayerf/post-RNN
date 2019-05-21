from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, SimpleRNN, Input
from keras.utils import to_categorical
from keras import optimizers
import numpy as np
import genDat as gd
import utils as ut
from keras import backend as K
from keras import callbacks
import matplotlib.pyplot as plt
from functools import partial
from params import *
from model_infrastructure import *

# For reproducebility
np.random.seed(1408)  
# Print parameters
print("Training on n= {:02d}, testing on n = {:03d}".format(train_n,test_n))
print("Prior variance is {:01.6f}".format(pow(sigma_theta,2)))
print("Posterior variance for training is {:01.6f}".format(pow(sigma_posterior,2)))
print("Posterior variance for testing is {:01.6f}".format(pow(t_sigma_posterior,2)))
print("Fitting RNN with the following architecture")


# Create Callbacks
my_average = average(t_batch_data,t_exact_quants,t_sigma_posterior, num_hidden)
# This callback will give loss vs. epoch in the logs
hist = callbacks.History()



# Give summary of architecture
print(model.summary(90))

# Initialize optimizer with given step size
adam = optimizers.adam(lr = step_size)
# Compile model w/ pinball loss, use mse as metric
model.compile(loss=pinball, metrics = ["mean_squared_error"],
              optimizer=adam)      
# Train the model using generator and callbacks we defined w/ test set as validation
history = model.fit_generator(genTraining(batch_size,train_n,sigma_theta),epochs=num_epochs,
                              steps_per_epoch=steps_per_epoch,
                              validation_data = (t_batch_data,t_batch_labels),
                              callbacks= [my_average,hist])
# Save test data
np.savetxt("labels.csv", t_batch_labels, delimiter=",")
np.savetxt("data.csv", t_batch_data[:,:,0], delimiter=",")
# Save miscoverage and predictions from test set.
np.savetxt("pb_mis",my_average.miscover)
np.savetxt("pb_avg",my_average.avg_miscover)
np.savetxt("average_preds", my_average.avg_model.predict(my_average.test))
np.savetxt("loss", hist.history['val_loss'])