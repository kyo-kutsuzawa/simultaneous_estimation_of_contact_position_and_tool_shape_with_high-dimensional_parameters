import numpy as np


class UnscentedKalmanFilter:
    def __init__(self, x0, x_dim, y_dim, P0, Q, R, kappa, alpha, beta, state_equation, observation_func, parallel):
        self.x_dim = x_dim
        self.y_dim = y_dim

        self.x = x0
        self.P = P0

        self.xa = np.zeros((parallel, self.x_dim*2+self.y_dim,))
        self.xa[:, :self.x_dim] = self.x

        self.Pa = np.zeros((parallel, self.x_dim*2+self.y_dim, self.x_dim*2+self.y_dim))
        self.Pa[:, 0:self.x_dim, 0:self.x_dim] = self.P
        self.Pa[:, self.x_dim:self.x_dim*2, self.x_dim:self.x_dim*2] = Q
        self.Pa[:, self.x_dim*2:self.x_dim*2+self.y_dim, self.x_dim*2:self.x_dim*2+self.y_dim] = R

        self.kappa = kappa
        self.alpha = alpha
        self.beta = beta

        self.state_equation = state_equation
        self.observation_func = observation_func

        self.parallel = parallel

    def reset(self, x, P, Q, R):
        self.x = x
        self.P = P

        self.xa = np.zeros((self.parallel, self.x_dim*2+self.y_dim,))
        self.xa[:, :self.x_dim] = self.x

        self.Pa = np.zeros((self.parallel, self.x_dim*2+self.y_dim, self.x_dim*2+self.y_dim))
        self.Pa[:, 0:self.x_dim, 0:self.x_dim] = self.P
        self.Pa[:, self.x_dim:self.x_dim*2, self.x_dim:self.x_dim*2] = Q
        self.Pa[:, self.x_dim*2:self.x_dim*2+self.y_dim, self.x_dim*2:self.x_dim*2+self.y_dim] = R

    def update(self, y, t, dt):
        # Calculate sigma points
        X, Wm, Wc = sample_scaled_sigma(self.xa, self.Pa, self.kappa, self.alpha, self.beta)
        Xx = X[:, :, 0:self.x_dim]
        Xv = X[:, :, self.x_dim:self.x_dim*2]
        Xn = X[:, :, self.x_dim*2:self.x_dim*2+self.y_dim]
        n_sigma = X.shape[1]

        # Time update
        # Update sigma points
        Xx = Xx.reshape(-1, self.x_dim)
        Xv = Xv.reshape(-1, self.x_dim)
        v = np.random.normal(size=Xx.shape) * Xv
        Xx = self.state_equation(Xx, v, t, dt)
        Xx = Xx.reshape(self.parallel, n_sigma, self.x_dim)
        Xv = Xv.reshape(self.parallel, n_sigma, self.x_dim)

        # Calculate mean of sigma points
        self.x = np.einsum("ij,ijk->ik", Wm, Xx)

        # Calculate variance of sigma points
        diff = Xx - self.x.reshape(-1, 1, self.x_dim)
        self.P = Wc.reshape(self.parallel, -1, 1, 1) * np.einsum("ijk,ijl->ijkl", diff, diff)
        self.P = np.sum(self.P, axis=1)

        # Calculate observation of sigma points
        Xx = Xx.reshape(-1, self.x_dim)
        Xn = Xn.reshape(-1, self.y_dim)
        v = np.random.normal(size=Xn.shape) * Xn
        Y = self.observation_func(Xx, y, v, t, dt)
        Xx = Xx.reshape(self.parallel, n_sigma, self.x_dim)
        Xn = Xn.reshape(self.parallel, n_sigma, self.y_dim)
        Y = Y.reshape(self.parallel, n_sigma, self.y_dim)

        # Calculate observation mean of sigma points
        self.y = np.einsum("ij,ijk->ik", Wm, Y)

        # Measurement update
        # Calculate observation variance of sigma points
        diff = Y - self.y.reshape(-1, 1, self.y_dim)
        Pyy = Wc.reshape(self.parallel, -1, 1, 1) * np.einsum("ijk,ijl->ijkl", diff, diff)
        Pyy = np.sum(Pyy, axis=1)

        # Calculate co-variance of sigma points
        diffx = Xx - self.x.reshape(-1, 1, self.x_dim)
        diffy = Y  - self.y.reshape(-1, 1, self.y_dim)
        Pxy = Wc.reshape(self.parallel, -1, 1, 1) * np.einsum("ijk,ijl->ijkl", diffx, diffy)
        Pxy = np.sum(Pxy, axis=1)

        # Calculate Kalman gain
        Pyy_inv = np.linalg.inv(Pyy)
        K = np.einsum("ijl,ilk->ijk", Pxy, Pyy_inv)

        # Update state mean and variance
        # self.x = self.x + np.einsum("ijk,ik->ij", K, y-self.y)  # Original
        self.x = self.x + np.einsum("ijk,ik->ij", K, y[2:]-self.y)  # Modified for contact estimation
        self.P = self.P - np.einsum("ijk,ikl,iml->ijm", K, Pyy, K)

        # Update total state and variance
        self.xa[:, 0:self.x_dim] = np.copy(self.x)
        self.Pa[:, 0:self.x_dim, 0:self.x_dim] = np.copy(self.P)

        return self.x


def sample_scaled_sigma(x, P, kappa, alpha, beta):
    # Setup constants
    parallel = x.shape[0]
    nx = x.shape[1]
    lam = alpha**2 * (nx + kappa) - nx

    # Setup variables
    X  = np.empty((parallel, 2*nx+1, nx))
    Wm = np.empty((parallel, 2*nx+1))
    Wc = np.empty((parallel, 2*nx+1))

    # Get N (with Cholesky Decomposition)
    N = np.linalg.cholesky((nx + lam)*P)

    # 0-th sigma point
    X[:, 0, :] = x
    Wm[:, 0] = lam / (nx + lam)
    Wc[:, 0] = lam / (nx + lam) + (1 - alpha**2 + beta)

    # 1-st to nx-th sigma point
    for i in range(1, nx+1):
        X[:, i, :] = x + N[:, i-1]
        Wm[:, i] = 1 / (2*(nx + lam))
        Wc[:, i] = 1 / (2*(nx + lam))

    # (nx+1)-th to (2*nx)-th sigma point
    for i in range(nx+1, nx*2+1):
        X[:, i, :] = x - N[:, i-(nx+1)]
        Wm[:, i] = 1 / (2*(nx + lam))
        Wc[:, i] = 1 / (2*(nx + lam))

    return X, Wm, Wc


def _example():
    from matplotlib import pyplot as plt

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
    P0 = np.identity(1).reshape(1, 1, 1) * 0.1
    Q  = np.identity(1).reshape(1, 1, 1) * 0.1
    R  = np.identity(1).reshape(1, 1, 1) * 0.1
    kappa = 0.0
    alpha = 0.9
    beta = 2.0
    ukf = UnscentedKalmanFilter(x, 1, 1, P0, Q, R, kappa, alpha, beta, state_equation, observation_func, N)

    Xest = np.empty((len(T), N, x.shape[1]))
    for i, (t, y) in enumerate(zip(T, Y)):
        y = np.tile(y, (N, 1))
        Xest[i, :, :] = ukf.update(y, t, dt)

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
