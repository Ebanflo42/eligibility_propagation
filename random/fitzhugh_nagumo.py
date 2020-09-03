import numpy as np
import tensorflow as tf


# Fitzhugh-Nagumo vector field
def vector_field(vw, a, b, tau, I):

    return np.array([vw[0] - vw[0]*vw[0]*vw[0]/3 - vw[1] + I, (vw[0] + a - b*vw[1])/tau])


# Generate trajectory
def simulate(initvw, eps, a, b, tau, noise_fcn, T):

    # trajectory in 2D phase space
    trajectory = np.zeros((T, 2))
    trajectory[0] += initvw

    # membrane potential is distance from w nullcline
    potentials = np.zeros(T)

    # Euler approximation
    for t in range(1, T):
        vw = trajectory[t - 1]
        trajectory[t] = vw + eps*vector_field(vw, a, b, tau, noise_fcn())
        potentials[t] = b*trajectory[t, 0] - trajectory[t, 1] - a

    spike_train = 0.5 + 0.5*np.sign(potentials)

    return trajectory, potentials, spike_train


# approximating firing rate, number of consecutive fires, and variance of hyper-theshold potential
# by assuming the step function for getting the spike train is a steep sigmoid
def metrics(potentials):

    T = float(potentials.shape.as_list()[0])

    spike_train_approx = tf.sigmoid(64.0*potentials)

    # always assuming that a time step is 1 millisecond
    count = tf.math.reduce_sum(spike_train_approx)
    rate = 1000.0*count/T

    # count consecutive spikes
    shifted_spikes = tf.roll(spike_train_approx, 1)
    repeats = tf.math.reduce_sum(spike_train*shifted_spikes)

    pos = tf.relu(potentials)
    mean = tf.math.reduce_sum(pos)/count
    centered = potentials - mean
    var = tf.math.reduce_sum(centered*centered)/(count - 1)

    return rate, repeats, var
