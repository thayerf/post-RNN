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
## Calulations, no need to edit
### PRIOR VAR IS ABOUT 25
np.random.seed(12345)

# Create Testing Data
t_dat = gd.gen_batch(t_batch_size)
junk = ut.setup_nn_mat(t_dat)
t_batch_labels = junk['outcome']


t_batch_data = junk['W']

# Create Callbacks
hist= callbacks.History()
print(t_batch_data[0,:,0])
# Print model summary
print(model.summary(90))
adam = optimizers.adam(lr = step_size, clipnorm= 0.5, clipvalue = 0.5)
model.compile(loss= pinball,optimizer=adam, metrics = ["mean_squared_error"])           
# Train the model
history = model.fit_generator(genTraining(batch_size),epochs=num_epochs,
                              steps_per_epoch=steps_per_epoch,
                              validation_data = (t_batch_data,t_batch_labels),
                              callbacks= [hist], verbose = 2)
# Save test data
np.savetxt("labels.csv", t_batch_labels, delimiter=",")
np.savetxt("I_data.csv", t_batch_data[:,:,0], delimiter=",")
np.savetxt("S_data.csv", t_batch_data[:,:,1], delimiter=",")

# Save test set preds
np.savetxt("preds", model.predict(t_batch_data))
# Save  model loss
np.savetxt("loss", hist.history['val_loss'])
# Save my model
model.save("SIR_model")
