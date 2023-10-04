import pdb
#%%Example 1 from https://github.com/sgvandijk/neural-processes/tree/master
import numpy as np
import random
import itertools
from tqdm import tqdm
import matplotlib.pyplot as plt
import tensorflow as tf
#Force tensorflow to use CPU as it's faster test example
tf.config.set_visible_devices([], 'GPU')

try:
    #Import from environment
    import neuralprocesses.tensorflow as nps
except:
    #Import local copy of neuralprocesses for debugging
    import sys
    sys.path.append('/Users/user/github/neuralprocesses')
    import neuralprocesses.tensorflow as nps

#Suppress tf warnings
tf.get_logger().setLevel('ERROR')

#Setup for plotting
ccycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
# Create an iterator that cycles through markers
marker_cycle = itertools.cycle(['o', 's', '^', 'D', 'x'])

def plot_samples(xs, sample_ys, xlim=(-1.0, 1.0), ax=None, c=None):
    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))
    if c is None:
        c = ccycle[0]
    n_samples = sample_ys.shape[1]
    for i in range(n_samples):
        ax.plot(xs, sample_ys[:,i], c=c, alpha=15.0 / n_samples)
    return ax


def split_context_target(xs, ys, n_context):
    """Randomly split a set of x,y samples into context and target sets"""
    context_mask = np.zeros(xs.shape[0], dtype=bool)
    context_mask[[i for i in random.sample(range(xs.shape[0]), n_context)]] = True

    context_xs = xs[context_mask]
    context_ys = ys[context_mask]

    target_xs = xs[~context_mask]
    target_ys = ys[~context_mask]

    return context_xs, context_ys, target_xs, target_ys


#%%Setup and demonstrate the data generator and demonstrate it
n_samples = 5
n_draws = 50

xs = np.linspace(-2, 2, n_samples).reshape([-1, 1]).astype(np.float32)
ys = np.sin(xs).astype(np.float32)

#n_iter is the number of times the data generator will run
#You run it like datagen = data_generator(10) for 10 batches of samples from the prior
#This data generator gives you a random number of data points from the prior xs,ys above
n_batches = 100
def data_generator(n_iter):
    for _ in range(n_iter):
        n_context = random.choice(range(1, n_samples))
        context_xs, context_ys, target_xs, target_ys = split_context_target(xs, ys, n_context)
        yield context_xs, context_ys, target_xs, target_ys
        
ds = tf.data.Dataset.from_generator(data_generator, output_types=(tf.float32, tf.float32, tf.float32, tf.float32), args=(n_batches,))


# #%% Plot the samples, setting up plot elements
# fig, ax = plt.subplots(figsize=(10,5))
# lns = [(plt.plot([], [], c=ccycle[0], alpha=0.05, animated=True))[0] for _ in range(n_draws)]
# target_sct = plt.scatter(xs, ys, c=ccycle[1])

# #%%Demonstrate the data generator sampling
# data_gen = data_generator(3)
# for context_xs, context_ys, target_xs, target_ys in data_gen:
#     plt.scatter(context_xs,context_ys,marker=next(marker_cycle),alpha=0.5)
    
    
#%% Setup and train the model
# Set up neural process
gnp = nps.construct_gnp(dim_x=1, dim_y=1, dim_lv=0, dim_embedding=2,
                        num_enc_layers=3,num_dec_layers=10,
                        width=16,
                        likelihood="lowrank",
                        enc_same=False) #This improves training if true

#Train settings
#(The concept of epochs doesn't really seem to make sense in this example
# because of how the data are generated)
n_epochs = 10
#n_train_draws = 100
n_test_draws = 200

#This is the test x data for testing the whole function, not just the training data points
test_xs = np.linspace(-10, 10, 100).reshape([-1, 1])

#This version of Adam works faster on a mac
optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=1e-3)
#optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

#A metric to keep track of the loss on a per-epoch basis
epoch_loss = tf.keras.metrics.Mean(name='train_loss')

#Predicted ys, to keep track of training
all_pred_ys = []
all_loss = []


#Test data as tf tensors of the right shape
xs_tf = tf.convert_to_tensor(xs.T)
xs_tf = tf.expand_dims(xs_tf,axis=0)
xs_tf = tf.expand_dims(xs_tf,axis=0)
ys_tf = tf.convert_to_tensor(ys.T)
ys_tf = tf.expand_dims(ys_tf,axis=0)
ys_tf = tf.expand_dims(ys_tf,axis=0)
test_xs_tf = tf.convert_to_tensor(test_xs.T)
test_xs_tf = tf.expand_dims(test_xs_tf,axis=0)
test_xs_tf = tf.expand_dims(test_xs_tf,axis=0)


for epoch in tqdm(range(1,n_epochs+1), desc="Training epochs"):
    if epoch == 0:
        #Start a new line so the first line of the output isn't messed up
        print("")
    
    #Reset the loss metric so we only have loss data from this epoch
    epoch_loss.reset_states()
        
    #The number of batches is controlled by ds
    #for context_xs, context_ys, target_xs, target_ys in ds:
    for xc, yc, xt, yt in ds:
        
        #At this point the data shapes are for example (3,1)
        
        #Make the dimensions work in gnp e.g. (1,1,3)
        xc = tf.transpose(xc)
        yc = tf.transpose(yc)
        xt = tf.transpose(xt)
        yt = tf.transpose(yt)
        xc = tf.expand_dims(xc, axis=0)
        yc = tf.expand_dims(yc, axis=0)
        xt = tf.expand_dims(xt, axis=0)
        yt = tf.expand_dims(yt, axis=0)
        
        # #Doesn't work, e.g. (3,1,1,1)
        # xc = tf.expand_dims(xc, axis=2)
        # yc = tf.expand_dims(yc, axis=2)
        # xt = tf.expand_dims(xt, axis=2)
        # yt = tf.expand_dims(yt, axis=2)
        # xc = tf.expand_dims(xc, axis=2)
        # yc = tf.expand_dims(yc, axis=2)
        # xt = tf.expand_dims(xt, axis=2)
        # yt = tf.expand_dims(yt, axis=2)


        with tf.GradientTape() as tape:                    
            # Compute the loss
            loss = -tf.reduce_mean(nps.loglik(gnp, xc,
                                   yc, xt, yt, normalise=True))
        
        #pdb.set_trace()
        # Compute gradients and update the model parameters.
        gradients = tape.gradient(loss, gnp.trainable_variables)
        optimizer.apply_gradients(zip(gradients, gnp.trainable_variables))
        #Update the loss metric
        epoch_loss(loss)
        
    # Make predictions with all known points as context
    mean, var, noiseless_samples, noisy_samples = nps.predict(gnp,xs_tf, ys_tf, test_xs_tf, num_samples=n_test_draws)
    all_pred_ys.append(tf.squeeze(noisy_samples))
    all_loss.append(epoch_loss.result())
    print(f"Loss: {epoch_loss.result()}")