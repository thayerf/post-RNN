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
from functools import partial

# For reproducebility
np.random.seed(1408)
## Define training data iterator
def genTraining(batch_size, train_n, sigma_theta):
    while True:
        # Get training data for step
        dat = gd.gen_batch(batch_size, train_n, sigma_theta)
        junk = ut.setup_nn_mat(dat)
        t = np.random.uniform(size = batch_size)
        t2 = np.repeat(t,train_n,axis = 0)
        t2 = t2.reshape(batch_size, train_n ,1)
        # We repeat the labels for each x in the sequence
        batch_labels = junk["outcome"]
        batch_data = junk["W"].reshape((batch_size, train_n, 1))
        batch_data = np.append(batch_data,t2,axis = 2)
        yield (
            [np.array(batch_data),np.array(t)],          
            np.array(batch_labels)
        )


# Create Callbacks
# my_average = average(t_batch_data,t_exact_quants,t_sigma_posterior, num_hidden, t_batch_labels)
# This callback will give loss vs. epoch in the logs
hist = callbacks.History()

# print("True loss: {:01.6f}".format(K.eval(pinball(my_average.exact,t_batch_labels))))


# Give summary of architecture

# Initialize optimizer with given step size
adam = optimizers.adam(lr=step_size, decay=step_size / num_epochs)
# Compile model w/ pinball loss, use mse as metric
model.compile(
    loss= partial(pinball, t),
    optimizer=adam,
)
print(model.summary(90))
# Train the model using generator and callbacks we defined w/ test set as validation
history = model.fit_generator(
    genTraining(batch_size, train_n, sigma_theta),
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=([t_batch_data,t_t], t_batch_labels),
    callbacks=[hist],
)
# Save test data
np.savetxt("labels.csv", t_batch_labels[:, 0], delimiter=",")
np.savetxt("data.csv", t_batch_data[:, :, 0], delimiter=",")
np.savetxt("quants.csv", t_t, delimiter = ",")
# Save miscoverage and predictions from test set.
# np.savetxt("pb_mis",my_average.miscover)
# np.savetxt("pb_avg",my_average.avg_miscover)
# np.savetxt("average_preds", my_average.avg_model.predict(my_average.test))
np.savetxt("loss", hist.history["val_loss"])
np.savetxt("preds", model.predict([t_batch_data,t_t]))
# np.savetxt("avg_loss", my_average.avg_loss)
