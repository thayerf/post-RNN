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
np.random.seed(1408)

print("Training on n= {:02d}, testing on n = {:02d}".format(train_n,test_n))
print("Prior variance is {:01.6f}".format(pow(sigma_theta,2)))
print("Fitting RNN with the following architecture")


# Create Callbacks
hist= callbacks.History()

# Print model summary
print(model.summary(90))
adam = optimizers.adam(lr = step_size, decay = decay_size)
model.compile(loss={'o1': pin_025, 'o2': pin_975},optimizer=adam)           
# Train the model on this epoch
history = model.fit_generator(genTraining(batch_size,train_n,sigma_theta),epochs=num_epochs,
                              steps_per_epoch=steps_per_epoch,
                              validation_data = (t_batch_data,[t_batch_labels,t_batch_labels]),
                              callbacks= [hist])
# Save test data
np.savetxt("CI_labels.csv", t_batch_labels, delimiter=",")
np.savetxt("CI_data.csv", t_batch_data[:,:,0], delimiter=",")
# Save test set preds
np.savetxt("CI_average_preds", model.predict(t_batch_data))

preds = model.predict(t_batch_data)
coverage = np.mean(np.equal(preds[:,0]<t_batch_labels,preds[:,2]> t_batch_labels))
width = np.mean(np.abs(preds[:,0]-preds[:,2]))
