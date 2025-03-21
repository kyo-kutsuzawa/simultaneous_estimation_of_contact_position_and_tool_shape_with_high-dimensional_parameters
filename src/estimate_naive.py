from dataclasses import dataclass
import enum
import os
import pickle
from subprocess import Popen, PIPE
import warnings
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tqdm
import tyro
from common import ShapeType, EstimationResult, EstimationMethod, NaiveParams
import myenv
from ukf import UnscentedKalmanFilter
import estimate_oracle as oracle

std_x = None
std_y = None
cell_size = None
shape_lim_x = None
shape_lim_y = None
std_f = 0.05
std_m = 0.005
encode = lambda x: x


class EncodeMethod(enum.Enum):
    sigmoid = enum.auto()
    softmax = enum.auto()


@dataclass
class Args:
    outdir: str = "result/result_naive.pickle"
    """Output directory"""
    load: str = ""
    """Filename of pre-recorded force signals if use"""
    plot: bool = True
    """Enable plotting"""
    shape: ShapeType = ShapeType.arch
    """Expected tool shape"""
    n_particles: int = 30
    """Number of particles"""
    cell_size: float = 0.4 / 2
    """Cell size of the tool-shape grid"""
    eff_th: float = 0.9
    """Threshold of the ratio of the effective number of particles"""
    std_c: float = 1e-4
    """Standard deviation of contact position used for estimation"""
    std_s: float = 1e-3
    """Standard deviation of contact position used for estimation"""
    std_y: float = 1e-2
    """Standard deviation of observed moment of force used for estimation"""
    initial_s_range: float = 1.0
    """Range of initial shape parameters: s ~ U(-initial_s_range, +initial_s_range)"""
    force_fluctuation_amp: float = -1.0
    """Amplitude of the contact force fluctuation; <0 for the default value"""
    save_all_shapes: bool = False
    """Save shape parameters at every timestep"""
    save_animation: bool = False
    """Save animation of estimation"""
    progressbar: bool = True
    """Show a progress bar"""
    encode: EncodeMethod = EncodeMethod.softmax
    """Encoding method for shoape parameters"""


def main(args: Args):
    global cell_size, std_x, std_y, shape_lim_x, shape_lim_y, encode

    cell_size = args.cell_size
    if args.encode == EncodeMethod.sigmoid:
        encode = sigmoid
    elif args.encode == EncodeMethod.softmax:
        encode = softmax

    # Initialization
    if args.load != "":
        shape_lim_x = (-0.2, 0.2)
        shape_lim_y = (-0.1, 0.3)
    else:
        shape_lim_x = (-0.0, 0.4)
        shape_lim_y = (-0.2, 0.2)

    n_size_x = int((shape_lim_x[1] - shape_lim_x[0]) / cell_size)
    n_size_y = int((shape_lim_y[1] - shape_lim_y[0]) / cell_size)
    x_dim = 2 + n_size_x * n_size_y
    y_dim = 1
    std_x = np.concatenate(
        [np.array([args.std_c, args.std_c]), np.full(n_size_x * n_size_y, args.std_s)],
        axis=0,
    )
    std_y = np.array([args.std_y])

    # Initialization
    if args.load != "":
        env = myenv.ContactEnvReal(args.load, shape_type=args.shape)

        def init_particle(n):
            _x0 = np.random.uniform(-0.1, 0.1, (n, 1))
            _x1 = np.random.uniform(0.0, 0.2, (n, 1))
            _s = np.random.normal(-0.1, 0.1, (n, n_size_x * n_size_y))
            return np.concatenate([_x0, _x1, _s], axis=1)

    else:
        env = myenv.ContactEnvSim(std_f=std_f, std_m=std_m, shape_type=args.shape)

        def init_particle(n, x=None):
            _x0 = np.random.uniform(0.0, 0.4, (n, 1))
            _x1 = np.random.uniform(-0.2, 0.2, (n, 1))
            if x is not None:
                _s = x[:, 2:]
            else:
                if args.initial_s_range > 0:
                    _s = np.random.uniform(
                        -args.initial_s_range,
                        args.initial_s_range,
                        (n, n_size_x * n_size_y),
                    )
                else:
                    _s = np.zeros((n, n_size_x * n_size_y))
            return np.concatenate([_x0, _x1, _s], axis=1)

        if args.force_fluctuation_amp >= 0:
            env.f_ang_amp = args.force_fluctuation_amp

    x = init_particle(args.n_particles)
    estimating = False

    ukf_args = {
        "x0": x,
        "x_dim": x_dim,
        "y_dim": y_dim,
        "P0": np.diag(std_x),
        "Q": np.diag(std_x),
        "R": np.diag(std_y),
        "kappa": 0.0,
        "alpha": 0.9,
        "beta": 2.0,
        "state_equation": state_equation,
        "observation_func": observation_func,
        "parallel": args.n_particles,
    }
    model = UnscentedKalmanFilter(**ukf_args)

    # Setup a figure
    shape_extent = (shape_lim_x[0], shape_lim_x[1], shape_lim_y[0], shape_lim_y[1])
    if args.plot or args.save_animation:
        fscale = 0.03
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_aspect("equal")
        ax1.set_xlim(shape_lim_x)
        ax1.set_ylim(shape_lim_y)
        env.plot(ax1, color="black", lw=3, solid_capstyle="round", zorder=1)
        s = x[:, 2:].reshape((-1, n_size_x, n_size_y))
        shape_plot = ax1.imshow(
            np.mean(encode(s), axis=0),
            vmin=0.0,
            vmax=1.0,
            origin="lower",
            extent=shape_extent,
        )
        positions = ax1.scatter(x[:, 0], x[:, 1], color="black", zorder=3)
        force = ax1.arrow(
            env.pc[0],
            env.pc[1],
            env.fc[0] * fscale,
            env.fc[1] * fscale,
            color="C3",
            width=0.003,
            zorder=2,
        )

        if args.load is not None:
            txt = plt.text(
                shape_lim_x[1] - 0.05,
                shape_lim_y[1] - 0.05,
                "t = ",
                fontsize="large",
                fontweight="bold",
                color="white",
            )
            ax1.set_xlabel("$y$ [m]")
            ax1.set_ylabel("$z$ [m]")
            ax1.invert_xaxis()
            frame_skip = 5
        else:
            txt = plt.text(
                shape_lim_x[0] + 0.05,
                shape_lim_y[1] - 0.05,
                "t = ",
                fontsize="large",
                fontweight="bold",
                color="white",
            )
            ax1.set_xlabel("$x$ [m]")
            ax1.set_ylabel("$y$ [m]")
            frame_skip = 10

        if args.save_animation:
            fps = 15.0
            ffmpeg_process = Popen(
                [
                    "ffmpeg",
                    "-y",  # Overwrite output files without asking
                    "-f",
                    "image2pipe",
                    "-r",  # Set frame rate (Hz value, fraction or abbreviation)
                    str(fps),
                    "-i",
                    "-",
                    "-r",
                    str(fps),
                    "-vcodec",
                    "h264",
                    "-pix_fmt",
                    "yuv420p",
                    "-b:v",  # Video bitrate
                    "2048k",
                    "-loglevel",
                    "warning",
                    os.path.join(os.path.dirname(args.outdir), "video.mp4"),
                ],
                stdin=PIPE,
            )
            write_frame(ffmpeg_process, fig)
            frame_skip = int((1.0 / fps) / env.dt)
            if frame_skip == 0:
                frame_skip = 1

    # Setup record
    params = NaiveParams(
        n_particles=args.n_particles,
        cell_size=cell_size,
        std_x=std_x,
        std_y=std_y,
        eff_th=args.eff_th,
        shape_extent=shape_extent,
    )
    record = EstimationResult(
        method=EstimationMethod.naive, shape_type=args.shape, hyper_params=params
    )

    # Setup a progress bar
    if args.progressbar:
        pbar = tqdm.tqdm(
            total=env.t_max,
            bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
        )
        warnings.simplefilter(
            "ignore", tqdm.TqdmWarning
        )  # Ignore warning when the progress-bar value exceeds 1

    # Estimation loop
    weights = np.full((args.n_particles,), 1 / args.n_particles)
    k = 0
    while not env.is_finished:
        # Get measurement values
        f, m = env.step()
        y = np.concatenate([f, m])

        if env.is_finished:
            break

        if np.linalg.norm(f) < 0.5:
            estimating = False
        else:
            if not estimating:
                x = init_particle(args.n_particles, x)
                ukf_args["x0"] = x.copy()
                model = UnscentedKalmanFilter(**ukf_args)
                estimating = True

            # Update UKF
            model.update(y, env.t, env.dt)

            # Sample
            v = np.random.normal(size=x.shape)
            x_next = np.einsum("ij,ijk->ik", v, model.P) + model.x

            # Weight
            w_pyx = pyx(y, x_next, env.t, env.dt)
            w_pxx = pxx(x_next, x, env.t, env.dt)
            w_q = gaussian_likelihood(x_next, model.x, model.P)
            weights *= w_pyx * w_pxx / w_q

            # Normalize weights
            weights = weights / np.sum(weights)

            # Resample particles
            n_eff = 1.0 / np.sum(weights**2)
            if n_eff < args.n_particles * args.eff_th:
                indices = np.array(list(range(args.n_particles)))
                indices = np.random.choice(indices, size=args.n_particles, p=weights)
                weights = np.full_like(weights, 1 / args.n_particles)

                x = x_next[indices, :]
                model.xa = model.xa[indices, :]
                model.Pa = model.Pa[indices, :, :]
            else:
                x = x_next

        # Record results
        record.pos_true.append(env.pc.copy())
        record.force_true.append(env.fc.copy())
        record.observations.append(np.concatenate([f, m]))
        record.pos_est.append(np.sum(weights.reshape(-1, 1) * x[:, :2], axis=0))
        if args.save_all_shapes:
            s = x[:, 2:].reshape((-1, n_size_x, n_size_y))
            record.shape_progress.append(np.mean(encode(s), axis=0))

        # Plot estimation
        if (args.plot or args.save_animation) and k % frame_skip == 0:
            x_show = x[:, 0:2].copy()

            s = x[:, 2:].reshape((-1, n_size_x, n_size_y))
            s_mean = np.mean(encode(s), axis=0)
            shape_plot.set_data(s_mean)
            shape_plot.set(clim=(np.min(s_mean), np.max(s_mean)))
            positions.set_offsets(x_show)
            force.set_data(
                x=env.pc[0], y=env.pc[1], dx=env.fc[0] * fscale, dy=env.fc[1] * fscale
            )
            txt.set_text("t = {:5.3f}".format(env.t))

            if args.plot:
                plt.pause(0.01)

            if args.save_animation:
                write_frame(ffmpeg_process, fig)

        k += 1
        if args.progressbar:
            pbar.update(env.dt)

    if args.progressbar:
        pbar.close()

    # Save the record
    s = x[:, 2:].reshape((-1, n_size_x, n_size_y))
    s_mean = np.mean(encode(s), axis=0)
    record.shape_est = s_mean
    if os.path.dirname(args.outdir) != "":
        os.makedirs(os.path.dirname(args.outdir), exist_ok=True)
    with open(args.outdir, "wb") as f:
        pickle.dump(record, f)

    if args.save_animation:
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x):
    z = np.exp(x)
    return z / (np.sum(z, axis=(1, 2), keepdims=True) * cell_size**2)


def gaussian_likelihood(x, mean, cov):
    dim = x.shape[1]
    coef = (2 * np.pi) ** dim * np.linalg.det(cov)
    coef = np.sqrt(coef)

    diff = x - mean
    cov_inv = np.linalg.inv(cov + np.identity(dim) * 1e-8)
    power = -0.5 * np.einsum("ij,ijk,ik->i", diff, cov_inv, diff)

    return np.exp(power) / coef


def state_equation(x, v, t, dt):
    x_next = np.copy(x) + v

    return x_next


def observation_func(x, y, v, t, dt):
    F = np.tile(y, (x.shape[0], 1))[:, 0:2]
    y_est = np.cross(x[:, 0:2], F).reshape((-1, 1)) + v

    return y_est


def pyx(y, x, t, dt):
    y_array = np.tile(y, (x.shape[0], 1))
    F = y_array[:, 0:2]
    M = y_array[:, 2:]
    mean = np.cross(x[:, 0:2], F).reshape((-1, 1))

    cov = np.diag(std_y)
    cov = np.tile(cov, (x.shape[0], 1, 1))

    return gaussian_likelihood(M, mean, cov)


def pxx(x_next, x, t, dt):
    mean = state_equation(x, 0, t, dt)
    cov = np.diag(std_x)
    cov = np.tile(cov, (x.shape[0], 1, 1))
    l1 = gaussian_likelihood(x_next, mean, cov)

    n_size_x = int((shape_lim_x[1] - shape_lim_x[0]) / cell_size)
    n_size_y = int((shape_lim_y[1] - shape_lim_y[0]) / cell_size)
    s = x[:, 2:].reshape((-1, n_size_x, n_size_y))

    idx = compute_grid_pos(x_next, s)
    l2 = encode(s)[idx]

    return l1 * l2


def compute_grid_pos(x, s):
    pos_lower_left = np.array([shape_lim_x[0], shape_lim_y[0]])
    normalized_pos = x[:, 0:2] - pos_lower_left
    idx = np.floor(normalized_pos / cell_size).astype(np.int64).T
    idx[0] = np.where(idx[0] < 0, 0, idx[0])
    idx[1] = np.where(idx[1] < 0, 0, idx[1])
    idx[0] = np.where(idx[0] >= s.shape[1], s.shape[1] - 1, idx[0])
    idx[1] = np.where(idx[1] >= s.shape[2], s.shape[2] - 1, idx[1])
    idx_tuple = (np.arange(s.shape[0]), idx[1], idx[0])

    return idx_tuple


def write_frame(p, fig):
    fig.canvas.draw()
    data = fig.canvas.tostring_rgb()
    width, height = fig.canvas.get_width_height()
    im = Image.frombytes("RGB", (width, height), data)
    im.save(p.stdin, "png")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
