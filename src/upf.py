import numpy as np
from ukf import UnscentedKalmanFilter


class UnscentedParticleFilter:
    def __init__(self, x0, x_dim, y_dim, P0, Q, R, kappa, alpha, beta, pyx, pxx, qxxy, state_equation, observation_func, n_particles):
        self.particles = x0
        self.weights = np.full((n_particles,), 1/n_particles)
        self.q = UnscentedKalmanFilter(self.particles, x_dim, y_dim, P0, Q, R, kappa, alpha, beta, state_equation, observation_func, n_particles)

        self.pyx = pyx
        self.pxx = pxx
        self.qxxy = qxxy

        self.n_particles = n_particles

    def reset(self, x, P, Q, R):
        pass

    def update(self, y, t, dt):
        # Update UKF
        y = np.tile(y, (self.n_particles, 1))
        self.q.update(y, t, dt)

        # Sample
        v = np.random.normal(size=self.particles.shape)
        self.particles_next = np.einsum("ij,ijk->ik", v, self.q.P) + self.q.x

        # Weight
        w_pyx = self.pyx(y, self.particles_next, t, dt)
        w_pxx = self.pxx(self.particles_next, self.particles, t, dt)
        w_q   = self.qxxy(self.particles_next, self.q)
        self.weights = w_pyx * w_pxx / w_q

        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)

    def resample(self):
        # Sample with replacement from the list of indices of the particles
        indices = np.array(list(range(self.n_particles)))
        indices = np.random.choice(indices, size=self.n_particles, p=self.weights)

        # Resample from particles
        self.particles = self.particles_next[indices, :]
        self.q.xa = self.q.xa[indices, :]
        self.q.Pa = self.q.Pa[indices, :, :]

        # Initialize weights
        self.weights = np.full_like(self.weights, 1/self.n_particles)

    def extract_sample(self):
        """Extract a set of estimation values from the particle group.
        """
        # Estimate contact position/force
        est = np.average(self.particles, axis=0, weights=self.weights)

        return est


def _example():
    from matplotlib import pyplot as plt

    def gaussian_likelihood(x, mean, cov):
        dim = x.shape[1]
        coef = (2*np.pi)**dim * np.linalg.det(cov)
        coef = np.sqrt(coef)

        diff = x - mean
        cov_inv = np.linalg.inv(cov + np.identity(dim)*1e-3)
        power = -0.5 * np.einsum("ij,ijk,ik->i", diff, cov_inv, diff)

        return np.exp(power) / coef

    def state_equation(x, v, t, dt):
        omega = 4e-2
        phi = 0.5

        x_next = 1 + np.sin(omega * np.pi * t) + phi * x + v
        return x_next

    def observation_func(x, v, t, dt):
        phi = 0.5

        if t <= 30:
            y = phi * x**2 + v
        else:
            y = phi * x - 2 + v
        return y

    def pyx(y, x, t, dt):
        std_y = np.array([0.1])

        mean = observation_func(x, 0, t, dt)

        cov = np.diag(std_y)
        cov = np.tile(cov, (x.shape[0], 1, 1))

        return gaussian_likelihood(y, mean, cov)

    def pxx(x_next, x, t, dt):
        std_x = np.array([0.1])

        mean = state_equation(x, 0, t, dt)

        cov = np.diag(std_x)
        cov = np.tile(cov, (x.shape[0], 1, 1))

        return gaussian_likelihood(x_next, mean, cov)

    def qxxy(particles_next, q):
        mean = q.x
        cov  = q.P

        l = gaussian_likelihood(particles_next, mean, cov)
        return l

    T = []
    X = []
    Y = []

    x = np.array([0.0])
    t = 0
    dt = 1.0
    while t < 60.0:
        vx = np.random.normal() * 0.3
        vy = np.random.normal() * 0.3
        x = state_equation(x, vx, t, dt)
        y = observation_func(x, vy, t, dt)

        X.append(x)
        Y.append(y)
        T.append(t)
        t += dt

    N = 10000
    x = np.zeros((N, 1))

    upf_args = {
        "x0": x,
        "x_dim": 1,
        "y_dim": 1,
        "P0": np.identity(1).reshape(1, 1, 1) * 0.1,
        "Q": np.identity(1).reshape(1, 1, 1) * 0.1,
        "R": np.identity(1).reshape(1, 1, 1) * 0.1,
        "kappa": 0.0,
        "alpha": 0.9,
        "beta": 2.0,
        "pyx": pyx,
        "pxx": pxx,
        "qxxy": qxxy,
        "state_equation": state_equation,
        "observation_func": observation_func,
        "n_particles": N
    }
    upf = UnscentedParticleFilter(**upf_args)

    Xest = np.empty((len(T), N, x.shape[1]))
    for i, (t, y) in enumerate(zip(T, Y)):
        upf.update(y, t, dt)
        upf.resample()
        Xest[i, :, :] = upf.particles

    Xest = np.array(Xest)
    Xest_mean = np.mean(Xest, axis=1).flatten()
    Xest_std  = np.std(Xest, axis=1).flatten()

    T = np.array(T).flatten()
    X = np.array(X).flatten()
    plt.scatter(T, X, color="b")
    plt.errorbar(T, Xest_mean, yerr=Xest_std, fmt="-o", color="r", ecolor="orange")
    plt.show()


if __name__ == "__main__":
    _example()
