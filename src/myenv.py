import random
import numpy as np
from scipy import optimize
from common import ShapeType


def shape_functions(shape_type):
    if shape_type == ShapeType.straight:
        _shape_func = lambda x: x * 0.0
        _shape_norm = lambda x: x * 0.0
    elif shape_type == ShapeType.arch:
        _shape_func = lambda x: -0.05 * np.cos(0.5 * np.pi * (x - 0.2) / 0.1)
        _shape_norm = lambda x: 0.05 * (0.5 * np.pi / 0.1) * np.sin(0.5 * np.pi * (x - 0.2) / 0.1)
    elif shape_type == ShapeType.angular:
        _shape_func = lambda x: -0.5 * (x - 0.2) * np.where(x > 0.2, 1, -1)
        _shape_norm = lambda x: np.where(x > 0.2, -0.5, 0.5)
    elif shape_type == ShapeType.wavy:
        _shape_func = lambda x: -0.05 * np.sin(0.5 * np.pi * (x - 0.2) / 0.08)
        _shape_norm = lambda x: -0.05 * (0.5 * np.pi / 0.08) * np.cos(0.5 * np.pi * (x - 0.2) / 0.08)
    elif shape_type == ShapeType.knife:
        _shape_func = lambda x: 0.001 * ((x / 0.1) ** 4 - 1)
        _shape_norm = lambda x: 4 * 0.001 / 0.1 * (x / 0.1) ** 3
    else:
        _shape_func = None
        _shape_norm = None

    return _shape_func, _shape_norm


def shape_functions_exp(shape_type):
    if shape_type == ShapeType.straight:
        _shape_func = lambda x: x * 0.0 - 0.013
        _shape_norm = lambda x: x * 0.0
    elif shape_type == ShapeType.angular:
        _shape_func = lambda x: 0.02 / 0.075 * (x - 0.085) * np.where(x > 0.085, 1, -1) - 0.045
        _shape_norm = lambda x: np.where(x > 0.085, 0.02 / 0.075, -0.02 / 0.075)
    elif shape_type == ShapeType.zigzag:
        _shape_func = lambda x: (0.03 / 0.05 * (x - 0.06) - 0.015) * np.where(x <= 0.06, 1, 0) \
            + (-0.03 / 0.05 * (x - 0.06) - 0.015) * np.where((0.06 < x) & (x <= 0.11), 1, 0) \
            + (0.035 / 0.05 * (x - 0.11) - 0.045) * np.where(x > 0.11, 1, 0)
        _shape_norm = lambda x: 0.03 / 0.05 * np.where(x <= 0.06, 1, 0) \
            - 0.03 / 0.05 * np.where((0.06 < x) & (x <= 0.11), 1, 0) \
            + 0.035 / 0.05 * np.where(x > 0.11, 1, 0)
    elif shape_type == ShapeType.discontinuous:
        _shape_func = lambda x: (0.02 / 0.06 * (x - 0.07) - 0.025) * np.where(x < 0.07, 1, 0) \
            + np.where((0.07 < x) & (x < 0.1), np.nan, 0) \
            + (-0.02 / 0.06 * (x - 0.1) - 0.025) * np.where(x > 0.1, 1, 0)
        _shape_norm = lambda x: 0.02 / 0.06 * np.where(x < 0.07, 1, 0) \
            - 0.02 / 0.06 * np.where(x > 0.1, 1, 0)

        # _shape_func = lambda x: (0.02 / 0.06 * (x - 0.07) - 0.025) * np.where(x < 0.07, 1, 0) \
        #     - 0.025 * np.where((0.07 < x) & (x < 0.1), 1, 0) \
        #     + (-0.02 / 0.06 * (x - 0.1) - 0.025) * np.where(x > 0.1, 1, 0)
        # _shape_norm = lambda x: 0.02 / 0.06 * np.where(x < 0.07, 1, 0) \
        #     - 0.02 / 0.06 * np.where(x > 0.1, 1, 0)
    else:
        _shape_func = None
        _shape_norm = None

    return _shape_func, _shape_norm


class ContactEnvSim:
    def __init__(self, std_f=0.0, std_m=0.0, f0=None, m0=None, shape_type="straight"):
        # Actual contact position & force
        self.pc = np.zeros((2,))
        self.fc = np.zeros((2,))

        # Time
        self.cnt = 0
        self.t = 0.0
        self.dt = 0.01
        self.t_max = 20.0
        self.interval = 100
        self.is_finished = False

        # Measurement noises
        self.std_f = std_f
        self.std_m = std_m

        # Measurement offset
        self.f0 = f0
        if f0 is None:
            self.f0 = np.zeros((2,))
        self.m0 = m0
        if m0 is None:
            self.m0 = np.zeros((1,))

        # fluctuation width of contact force
        self.f_ang_amp = np.pi * 0.167

        # Shape settings
        self.x_range = (0.1, 0.3)
        self._shape_func, self._shape_norm = shape_functions(shape_type)

        n_contact = 10
        self.x_pos = np.linspace(*self.x_range, n_contact).tolist()
        random.shuffle(self.x_pos)
        self.fluctuate = True

    def step(self):
        if self.cnt % self.interval == 0:
            if len(self.x_pos) > 0:
                self.pc[0] = self.x_pos.pop()
            else:
                self.pc[0] = np.random.uniform(*self.x_range)
                self.fluctuate = False
            self.pc[1] = self.shape_func(self.pc[0])
            self.th_tangent = np.arctan2(1, self.shape_norm(self.pc[0])) - np.pi * 0.5
            self.f_amp = np.random.uniform(1.0, 3.0)
            self.fc[0] = 0.0
            self.fc[1] = 0.0
            if not self.fluctuate:
                self.f_ang = self.f_ang_amp * np.random.uniform(-1.0, 1.0) + self.th_tangent
        else:
            if self.fluctuate:
                self.f_ang = self.f_ang_amp * np.sin(2 * np.pi * 2.0 * self.t) + self.th_tangent
            self.fc[0] = self.f_amp * np.sin(self.f_ang)
            self.fc[1] = self.f_amp * np.cos(self.f_ang)

        # Calculate measurement values
        f = np.copy(self.fc)
        m = np.cross(self.pc, self.fc).reshape(1)

        f += self.f0 + np.random.normal(size=f.shape) * self.std_f
        m += self.m0 + np.random.normal(size=m.shape) * self.std_m

        self.cnt += 1
        self.t = self.cnt * self.dt
        if self.t > self.t_max:
            self.is_finished = True

        return f, m

    def step_bak(self):
        if self.cnt < 7500:
            if self.cnt % 100 == 0:
                # Update contact position
                self.pc[0] = np.random.uniform(*self.x_range)
                self.pc[1] = self.shape_func(self.pc[0])

                # Update contact force
                self.th_tangent = np.arctan2(1, self.shape_norm(self.pc[0])) - np.pi * 0.5
                self.f_amp = np.random.uniform(1.0, 3.0)

            self.f_ang = self.f_ang_amp * np.sin(2 * np.pi * 2.0 * self.t) + self.th_tangent
        else:
            if self.cnt % 100 == 0:
                # Update contact position
                self.pc[0] = np.random.uniform(*self.x_range)
                self.pc[1] = self.shape_func(self.pc[0])

                # Update contact force
                th_tangent = np.arctan2(1, self.shape_norm(self.pc[0])) - np.pi * 0.5
                self.f_amp = np.random.uniform(1.0, 3.0)
                self.f_ang = self.f_ang_amp * np.random.uniform(-1.0, 1.0) + th_tangent

        self.fc[0] = self.f_amp * np.sin(self.f_ang)
        self.fc[1] = self.f_amp * np.cos(self.f_ang)

        if self.cnt % 100 == 0:
            self.fc[0] = 0.0
            self.fc[1] = 0.0

        # Calculate measurement values
        f = np.copy(self.fc)
        m = np.cross(self.pc, self.fc).reshape(1)

        f += self.f0 + np.random.normal(size=f.shape) * self.std_f
        m += self.m0 + np.random.normal(size=m.shape) * self.std_m

        self.cnt += 1
        self.t = self.cnt * self.dt
        if self.t > self.t_max:
            self.is_finished = True

        return f, m

    def shape_func(self, x):
        x_clip = np.clip(x, *self.x_range)
        y = self._shape_func(x_clip)
        return y

    def shape_norm(self, x):
        y = self._shape_norm(x)
        return y

    def plot(self, ax, **kwargs):
        x = np.linspace(*self.x_range, 100)
        y = self.shape_func(x)
        ax.plot(x, y, **kwargs)


class ContactEnvReal:
    def __init__(self, filename="", calibration_duration=5000, shape_type="straight"):
        if filename !="":
            rec = np.loadtxt(filename, delimiter=",")
            self.data = rec[:, (2, 3, 4)]  # y- and z-axes force and x-axis moment
        else:
            self.data = np.zeros((100, 3))

        # Actual contact position & force
        self.pc = np.zeros((2,))
        self.fc = np.zeros((2,))

        # Time
        self.cnt = calibration_duration
        self.skip_interval = 20
        self.dt = 0.001 * self.skip_interval
        self.t = self.cnt / self.skip_interval * self.dt
        self.t_max = (self.data.shape[0] - calibration_duration) / self.skip_interval * self.dt
        self.is_finished = False

        # Measurement offset
        self.f0 = np.mean(self.data[:calibration_duration, :2], axis=0)
        self.m0 = np.mean(self.data[:calibration_duration, 2:], axis=0)

        # Shape
        self.x_range = (0.01, 0.16)
        self._shape_func, self._shape_norm = shape_functions_exp(shape_type)

    def step(self):
        f = self.data[self.cnt, :2] - self.f0
        m = self.data[self.cnt, 2:] - self.m0

        self.fc = f
        if np.linalg.norm(f) > 0.5:
            self.pc = self._estimate_contact_position(f, m)
        else:
            self.pc = np.full((2,), np.nan)

        self.cnt += self.skip_interval
        self.t = self.cnt / self.skip_interval * self.dt
        if self.cnt >= self.data.shape[0]:
            self.is_finished = True

        return f, m

    def shape_func(self, x):
        x_clip = np.clip(x, *self.x_range)
        y = self._shape_func(x_clip)
        return y

    def plot(self, ax, **kwargs):
        z = np.linspace(*self.x_range, 100)
        y = self.shape_func(z)
        ax.plot(y, z, **kwargs)


    def _estimate_contact_position(self, force, moment):
        # Compute contact-position candidates
        f_norm = np.dot(force, force)
        cx = lambda alpha: alpha * force[0] + force[1] * moment / f_norm
        cy = lambda alpha: alpha * force[1] - force[0] * moment / f_norm

        # Solve an optimization problem
        obj_func = lambda alpha: (cx(alpha) - self.shape_func(cy(alpha))) ** 2

        # Compute the optimal alpha while its error is less than 10^{-10}
        value = 1.0
        while value > 1e-10:
            x0 = np.random.uniform(-1, 1)
            res = optimize.minimize(obj_func, x0, tol=1e-10)
            value = res.fun

        # Compute the estimated contact position
        p_estimate = np.array([float(cx(res.x)), float(cy(res.x))])

        return p_estimate


def _example():
    # Setup contact environment
    env = ContactEnvSim()

    # Initialize records
    F  = []
    M  = []
    Pc = []
    Fc = []
    T  = []

    while env.t < 10.0:
        # Measure F/T values
        f, m = env.step()

        # Record values
        F.append(f)
        M.append(m)
        Pc.append(np.copy(env.pc))
        Fc.append(np.copy(env.fc))
        T.append(env.t)

    # Convert records to numpy.ndarray
    F = np.stack(F)
    M = np.stack(M)
    Pc = np.stack(Pc)
    Fc = np.stack(Fc)
    T = np.stack(T)

    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax1 = fig.add_subplot(2, 2, 1)
    ax2 = fig.add_subplot(2, 2, 2)
    ax3 = fig.add_subplot(2, 2, 3)
    ax4 = fig.add_subplot(2, 2, 4)

    # Plot measured force values
    ax1.plot(T, F[:, 0], label="x")
    ax1.plot(T, F[:, 1], label="y")
    ax1.set_xlim((T[0], T[-1]))
    ax1.set_ylabel("Measured force [N]")

    # Plot measured moment values
    ax2.plot(T, M[:], label="z")
    ax2.set_xlim((T[0], T[-1]))
    ax2.set_ylabel("Measured moment [Nm]")

    # Plot measured force values
    ax3.plot(T, Pc[:, 0], label="x")
    ax3.plot(T, Pc[:, 1], label="y")
    ax3.set_xlim((T[0], T[-1]))
    ax3.set_ylabel("Contact position [m]")

    # Plot measured moment values
    ax4.plot(T, Fc[:, 0], label="x")
    ax4.plot(T, Fc[:, 1], label="y")
    ax4.set_xlim((T[0], T[-1]))
    ax4.set_ylabel("Contact force [N]")

    plt.show()


def _example_shape():
    import matplotlib.pyplot as plt

    _, ax = plt.subplots(1, 1)
    ax.set_xlim(-0.0, 0.4)
    ax.set_ylim(-0.2, 0.2)
    ax.set_aspect("equal")

    x = np.linspace(0.1, 0.3, 1000)
    ax.plot(x, shape_functions(ShapeType.straight)[0](x))
    ax.plot(x, shape_functions(ShapeType.arch)[0](x))
    ax.plot(x, shape_functions(ShapeType.angular)[0](x))
    ax.plot(x, shape_functions(ShapeType.wavy)[0](x))
    ax.plot(x, shape_functions(ShapeType.knife)[0](x))

    plt.show()


if __name__ == "__main__":
    from matplotlib import pyplot as plt

    _example()
