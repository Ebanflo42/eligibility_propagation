import numpy as np

# Fitzhugh-Nagumo vector field
def vector_field(vw, a, b, tau, I):

    v, w = vw[0], vw[1]

    vdot = v - 0.333*v*v*v - w + I
    wdot = (v + a - b*w)/tau

    return vdot, wdot

# Generate trajectory
def simulate(initvw, eps, a, b, tau, noise_fcn, T):

    # spike train is determined by which side of the nullcline the system is on
    # return whether or not there is a spike and the potential
    def test_w_nullcline(vw):
        d = b*vw[0] - vw[1]
        u = d - a
        if d < 0:
            return 0, d
        else:
            return 1, d

    # trajectory in 2D phase space
    trajectory = np.zeros((T, 2))
    trajectory[0] = np.array(initvw)

    # potential and spike train
    spike_train = np.zeros(T)
    potentials = np.zeros(T)
    spike_train[0], potentials[0] = test_w_nullcline(trajectory[0])

    # Euler approximation
    for t in range(1, T):
        vw = trajectory[t - 1]
        trajectory[t] = vw + eps*np.array(vector_field(vw, a, b, tau, noise_fcn()))
        spike_train[t], potentials[t] = test_w_nullcline(trajectory[t])

    # determine firing rate and the number of repeated fires
    rate, n = 0, 0
    rate += spike_train[0]
    for i in range(1, T):
        rate += spike_train[i]
        n += spike_train[i]*spike_train[i - 1]
    rate /= T

    spike_potentials = []
    for v in potentials:
        if v > 0:
            spike_potentials.append(v)
    v = np.std(spike_potentials)

    return trajectory, spike_train, rate, n, v