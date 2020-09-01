import numpy as np

def vector_field(vw, a, b, tau, I):

    v, w = vw[0], vw[1]

    vdot = v - 0.333*v*v*v - w + I
    wdot = (v + a - b*w)/tau

    return vdot, wdot


def simulate(initvw, eps, a, b, tau, noise_fcn, T):

    def test_w_nullcline(vw):
        d = b*vw[0] - vw[1]
        #print(d)
        u = d - a
        if d < 1:
            return 0
        else:
            return d

    trajectory = np.zeros((T, 2))
    trajectory[0] = np.array(initvw)

    spike_train = np.zeros(T)
    spike_train[0] = test_w_nullcline(trajectory[0])

    for t in range(1, T):
        vw = trajectory[t - 1]
        trajectory[t] = vw + eps*np.array(vector_field(vw, a, b, tau, noise_fcn()))
        spike_train[t] = test_w_nullcline(trajectory[t])

    return trajectory, spike_train