import numpy as np
import tensorflow as tf


# Fitzhugh-Nagumo vector field
def vector_field(vw, a, b, tau, I):

    v, w = vw[0], vw[1]

    vdot = v - 0.333*v*v*v - w + I
    wdot = (v + a - b*w)/tau

    return tf.constant([vdot, wdot])


# Generate trajectory
def simulate(initvw, eps, a, b, tau, noise_fcn, T):

    # trajectory in 2D phase space
    trajectory = tf.zeros((T, 2))
    trajectory[0] = tf.constant(initvw)

    # membrane potential is distance from w nullcline
    potentials = tf.zeros(T)

    # Euler approximation
    for t in range(1, T):
        vw = trajectory[t - 1]
        trajectory[t] = vw + eps*vector_field(vw, a, b, tau, noise_fcn())
        potentials[t] = b*trajectory[t, 0] - trajectory[t, 1] - a

    spike_train = 0.5 + 0.5*tf.sign(potentials)

    return trajectory, potentials, spike_train


# approximating firing rate, number of consecutive fires, and variance of hyper-theshold potential
# by assuming the step function for getting the spike train is a steep sigmoid
def metrics(potentials):

    T = len(potentials)

    spike_train_approx = tf.sigmoid(64*potentials)

    # always assuming that a time step is 1 millisecond
    count = tf.sum(spike_train_approx)
    rate = 1000*count/T

    repeats = tf.constant([0])

    for t in range(1, T):
        repeats += spike_train_approx[t]*spike_train_approx[t - 1]

    pos = tf.relu(potentials)
    mean = tf.sum(pos)/count
    shifted = potentials - mean
    var = tf.sum(shifted*shifted)/(count - 1)

    return rate, repeats, var
