from keras.models import Sequential
from keras.layers import LSTM, Dense, TimeDistributed, SimpleRNN, Input
from keras.utils import to_categorical
from keras import optimizers
import numpy as np
import genDat as gd
import utils as ut
from keras import backend as K
from keras import callbacks
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
print("Risk of true minimizer is {:01.6f}".format(base_risk))
print("Fitting RNN with the following architecture")

test_n = sys.argv[1]
train_n = sys.argv[1]
# Create Callbacks
#my_average = average(t_batch_data,t_exact_quants,t_sigma_posterior, num_hidden, t_batch_labels)
# This callback will give loss vs. epoch in the logs
hist = callbacks.History()

#print("True loss: {:01.6f}".format(K.eval(pinball(my_average.exact,t_batch_labels))))


# Give summary of architecture
print(model.summary(90))

# Initialize optimizer with given step size
adam = optimizers.adam(lr = step_size, decay = step_size/num_epochs)
# Compile model w/ pinball loss, use mse as metric
model.compile(loss=pinball,
              optimizer=adam)      
# Train the model using generator and callbacks we defined w/ test set as validation
history = model.fit_generator(genTraining(batch_size,train_n,sigma_theta),epochs=num_epochs,
                              steps_per_epoch=steps_per_epoch,
                              validation_data = (t_batch_data,t_batch_labels),
                              callbacks= [hist])
# Save test data
np.savetxt("labels.csv", t_batch_labels, delimiter=",")
np.savetxt("data.csv", t_batch_data[:,:,0], delimiter=",")
# Save predictions and loss
np.savetxt("loss", hist.history['val_loss'])
model.save("my_model")
np.savetxt('preds',model.predict(t_batch_data)[:,:,0])