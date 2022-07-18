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
import matplotlib.pyplot as plt
# For reproducebility
np.random.seed(1408)
# Print parameters
print("Training on n= {:02d}, testing on n = {:03d}".format(train_n, test_n))
print("Prior variance is {:01.6f}".format(pow(sigma_theta, 2)))
print("Posterior variance for training is {:01.6f}".format(pow(sigma_posterior, 2)))
print("Posterior variance for testing is {:01.6f}".format(pow(t_sigma_posterior, 2)))
print("Fitting RNN with the following architecture")
## Define training data iterator
def genTraining(batch_size, train_n, sigma_theta):
    while True:
        # Get training data for step
        dat = gd.gen_batch(batch_size, train_n, sigma_theta)
        junk = ut.setup_nn_mat(dat)
        # We repeat the labels for each x in the sequence
        batch_labels = junk["outcome"]
        batch_data = junk["W"].reshape((batch_size, train_n, 1))
        yield (
            np.array(batch_data),
            [
                np.array(batch_labels),
                np.array(batch_labels),
                np.array(batch_labels),
                np.array(batch_labels),
                np.array(batch_labels),
                np.array(batch_labels),
                np.array(batch_labels),
                np.array(batch_labels),
                np.array(batch_labels),
            ],
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
    loss=[
        partial(pinball, 0.1),
        partial(pinball, 0.2),
        partial(pinball, 0.3),
        partial(pinball, 0.4),
        partial(pinball, 0.5),
        partial(pinball, 0.6),
        partial(pinball, 0.7),
        partial(pinball, 0.8),
        partial(pinball, 0.9),
    ],
    optimizer=adam,
    loss_weights=[1, 1, 1, 1, 1, 1, 1, 1, 1],
)
print(model.summary(90))
# Train the model using generator and callbacks we defined w/ test set as validation
history = model.fit_generator(
    genTraining(batch_size, train_n, sigma_theta),
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=(t_batch_data, t_batch_labels),
    callbacks=[hist],
)
# Save test data
np.savetxt("labels.csv", t_batch_labels[0][:, 0], delimiter=",")
np.savetxt("data.csv", t_batch_data[:, :, 0], delimiter=",")
# Save miscoverage and predictions from test set.
# np.savetxt("pb_mis",my_average.miscover)
# np.savetxt("pb_avg",my_average.avg_miscover)
np.savetxt("preds", np.squeeze(model.predict(t_batch_data),axis = 2))
np.savetxt("loss", hist.history["val_loss"])
# np.savetxt("avg_loss", my_average.avg_loss)
