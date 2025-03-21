from dataclasses import dataclass
import enum
import os
import pickle
from subprocess import Popen, PIPE
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tqdm
import tyro
from common import ShapeType, EstimationResult, EstimationMethod, ProposedParams
import myenv
from ukf import UnscentedKalmanFilter

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
    outdir: str = "result/result_proposed.pickle"
    """Output directory"""
    load: str = ""
    """Filename of pre-recorded force signals if use"""
    plot: bool = True
    """Enable plotting"""
    shape: ShapeType = ShapeType.arch
    """Expected tool shape"""
    n_particles: int = 30
    """Number of particles"""
    cell_size: float = 0.005
    """Cell size of the tool-shape grid"""
    d_th: float = 0.006
    """Hyperparameter for shape estimation"""
    theta_th: float = np.pi / 10
    """Hyperparameter for shape estimation"""
    s_inc: float = 0.064
    """Hyperparameter for shape estimation"""
    s_dec: float = 0.0063
    """Hyperparameter for shape estimation"""
    eff_th: float = 0.9
    """Threshold of the ratio of the effective number of particles"""
    std_x: float = 4e-5
    """Standard deviation of contact position used for estimation"""
    std_y: float = 3e-4
    """Standard deviation of observed moment of force used for estimation"""
    force_fluctuation_amp: float = -1.0
    """Amplitude of the contact force fluctuation; <0 for the default value"""
    save_all_shapes: bool = False
    """Save shape parameters at every timestep"""
    use_true_shape: bool = False
    """Use the true shape as the initial values of the estimated shape parameters"""
    save_animation: bool = False
    """Save animation of estimation"""
    progressbar: bool = True
    """Show a progress bar"""
    encode: EncodeMethod = EncodeMethod.softmax
    """Encoding method for shoape parameters"""
    use_fb: bool = False
    """Plot frame buffer"""


def main(args: Args):
    global cell_size, std_x, std_y, shape_lim_x, shape_lim_y, encode

    cell_size = args.cell_size
    std_x = np.array([args.std_x, args.std_x])
    std_y = np.array([args.std_y])
    if args.encode == EncodeMethod.sigmoid:
        encode = sigmoid
    elif args.encode == EncodeMethod.softmax:
        encode = softmax

    if os.path.dirname(args.outdir) != "":
        os.makedirs(os.path.dirname(args.outdir), exist_ok=True)

    # Initialization
    if args.load != "":
        env = myenv.ContactEnvReal(args.load, shape_type=args.shape)
        shape_lim_x = (-0.2, 0.2)
        shape_lim_y = (-0.1, 0.3)

        def init_particle(n):
            _x0 = np.random.uniform(-0.1, 0.1, (n,))
            _x1 = np.random.uniform(0.0, 0.2, (n,))
            return np.stack([_x0, _x1], axis=1)

    else:
        env = myenv.ContactEnvSim(std_f=std_f, std_m=std_m, shape_type=args.shape)
        shape_lim_x = (-0.0, 0.4)
        shape_lim_y = (-0.2, 0.2)

        def init_particle(n):
            _x0 = np.random.uniform(-0.0, 0.4, (n,))
            _x1 = np.random.uniform(-0.2, 0.2, (n,))
            return np.stack([_x0, _x1], axis=1)

        if args.force_fluctuation_amp >= 0:
            env.f_ang_amp = args.force_fluctuation_amp

    x_dim = 2
    y_dim = 1
    n_size_x = int((shape_lim_x[1] - shape_lim_x[0]) / cell_size)
    n_size_y = int((shape_lim_y[1] - shape_lim_y[0]) / cell_size)

    x = init_particle(args.n_particles)
    s = np.full((args.n_particles, n_size_x, n_size_y), 0.0)
    estimating = False

    if args.use_true_shape:
        s = get_true_shape(s, env)

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
        shape_plot = ax1.imshow(
            np.mean(encode(s), axis=0),
            vmin=0.0,
            vmax=1.0,
            origin="lower",
            extent=shape_extent,
            zorder=0,
        )
        positions = ax1.scatter(x[:, 0], x[:, 1], color="black", zorder=3)
        pos_est = ax1.scatter(
            np.mean(x[:, 0]), np.mean(x[:, 1]), color="pink", zorder=4
        )
        force = ax1.arrow(
            env.pc[0],
            env.pc[1],
            env.fc[0] * fscale,
            env.fc[1] * fscale,
            color="C3",
            width=0.003,
            zorder=2,
        )

        if args.load != "":
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
            fps = 25.0
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

        if args.use_fb:
            import framebuffer
            fb = framebuffer.FrameBuffer()

    # Setup record
    params = ProposedParams(
        n_particles=args.n_particles,
        cell_size=cell_size,
        d_th=args.d_th,
        theta_th=args.theta_th,
        s_inc=args.s_inc,
        s_dec=args.s_dec,
        std_x=std_x,
        std_y=std_y,
        eff_th=args.eff_th,
        shape_extent=shape_extent,
    )
    record = EstimationResult(
        method=EstimationMethod.proposed,
        shape_type=args.shape,
        hyper_params=params,
    )
    if args.force_fluctuation_amp >= 0:
        record.force_fluctuation_amp = args.force_fluctuation_amp

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
    while True:
        # Get measurement values
        f, m = env.step()
        y = np.concatenate([f, m])

        if env.is_finished:
            break

        if np.linalg.norm(f) < 0.5:
            estimating = False
        else:
            if not estimating:
                x = init_particle(args.n_particles)
                ukf_args["x0"] = x.copy()
                model = UnscentedKalmanFilter(**ukf_args)
                estimating = True

            # Update UKF
            model.update(y, env.t, env.dt)

            # Sample
            v = np.random.normal(size=x.shape)
            if not args.use_true_shape:
                s = update_shape(
                    s,
                    x,
                    y,
                    theta=args.theta_th,
                    near_th=args.d_th,
                    s_inc=args.s_inc,
                    s_dec=args.s_dec,
                )
            x_next = np.einsum("ij,ijk->ik", v, model.P) + model.x

            # Weight
            w_pyx = pyx(y, x_next, env.t, env.dt)
            w_pxx = pxx(x_next, x, s, env.t, env.dt)
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
                s = s[indices, :, :]
                model.xa = model.xa[indices, :]
                model.Pa = model.Pa[indices, :, :]
            else:
                x = x_next

        # Record results
        record.pos_true.append(env.pc.copy())
        record.force_true.append(env.fc.copy())
        record.observations.append(np.concatenate([f, m]))
        record.pos_est.append(np.sum(weights.reshape(-1, 1) * x, axis=0))
        if args.save_all_shapes:
            record.shape_progress.append(np.mean(encode(s), axis=0))

        # Plot estimation
        if (args.plot or args.save_animation) and k % frame_skip == 0:
            x_show = x[:, 0:2].copy()
            if not estimating:
                x_show = np.full_like(x[:, 0:2], np.nan)

            s_mean = np.mean(encode(s), axis=0)
            shape_plot.set_data(s_mean * n_size_x * n_size_y * cell_size**2 * 0.5)
            # shape_plot.set(clim=(np.min(s_mean), np.max(s_mean)))
            positions.set_offsets(x_show)
            pos_est.set_offsets(np.sum(weights.reshape(-1, 1) * x_show, axis=0))
            if estimating:
                force.set_data(
                    x=env.pc[0],
                    y=env.pc[1],
                    dx=env.fc[0] * fscale,
                    dy=env.fc[1] * fscale,
                )
            else:
                force.set_data(
                    x=1.0, y=1.0, dx=env.fc[0] * fscale, dy=env.fc[1] * fscale
                )
            txt.set_text("t = {:5.3f}".format(env.t))

            if args.plot:
                if args.use_fb:
                    fig.canvas.draw()
                    fb.show(fig.canvas)
                    time.sleep(0.01)
                else:
                    plt.pause(0.01)

            if args.save_animation:
                write_frame(ffmpeg_process, fig)

        k += 1
        if args.progressbar:
            pbar.update(env.dt)

    if args.progressbar:
        pbar.close()

    # Save the record
    record.shape_est = np.mean(encode(s), axis=0)
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
    y_est = np.cross(x, F).reshape((-1, 1)) + v

    return y_est


def pyx(y, x, t, dt):
    y_array = np.tile(y, (x.shape[0], 1))
    F = y_array[:, 0:2]
    M = y_array[:, 2:]
    mean = np.cross(x, F).reshape((-1, 1))

    cov = np.diag(std_y)
    cov = np.tile(cov, (x.shape[0], 1, 1))

    return gaussian_likelihood(M, mean, cov)


def pxx(x_next, x, s, t, dt):
    mean = state_equation(x, 0, t, dt)
    cov = np.diag(std_x)
    cov = np.tile(cov, (x.shape[0], 1, 1))
    l1 = gaussian_likelihood(x_next, mean, cov)

    idx = compute_grid_pos(x_next, s)
    l2 = encode(s)[idx]

    return l1 * l2


def update_shape(s, x, y, theta=np.pi / 6, near_th=0.1, s_inc=0.01, s_dec=0.01):
    n_particles = s.shape[0]

    c = x[:, 0:2]
    f = np.tile(y, (x.shape[0], 1))[:, 0:2]
    nf = f / np.linalg.norm(f, axis=1, keepdims=True)

    grid_size_x = s.shape[1]
    grid_size_y = s.shape[2]
    center_points_x = np.linspace(*shape_lim_x, grid_size_x)
    center_points_y = np.linspace(*shape_lim_y, grid_size_y)
    p_cells = np.meshgrid(center_points_x, center_points_y)
    p_cells = np.stack(p_cells)

    for i in range(n_particles):
        rel_pos = p_cells - c[i].reshape(-1, 1, 1)
        norm_rel_pos = np.linalg.norm(rel_pos, axis=0, keepdims=True)
        npos = rel_pos / norm_rel_pos

        products = np.einsum("ijk,i->jk", npos, nf[i])
        cone_condition = abs(products) > np.cos(theta)
        near_condition = norm_rel_pos[0] < near_th
        s[i, :, :] += np.where(cone_condition, -s_dec, 0.0)
        s[i, :, :] += np.where(near_condition, +s_inc, 0.0)

    s = np.clip(s, -5.0, 5.0)

    return s


def compute_grid_pos(x, s):
    pos_lower_left = np.array([shape_lim_x[0], shape_lim_y[0]])
    normalized_pos = x[:, 0:2] - pos_lower_left
    idx = np.floor(normalized_pos / cell_size).astype(np.int64).T
    idx[0] = np.clip(idx[0], 0, s.shape[1] - 1)
    idx[1] = np.clip(idx[1], 0, s.shape[2] - 1)
    idx_tuple = (np.arange(s.shape[0]), idx[1], idx[0])

    return idx_tuple


def get_true_shape(s, env):
    x = np.linspace(*env.x_range, 100)
    y = env.shape_func(x)

    pos = np.stack([x, y], axis=1)
    pos = pos.reshape(-1, 1, 2)
    pos = np.tile(pos, (1, s.shape[0], 1))

    s[:, :, :] = -np.inf

    for p in pos:
        idx = compute_grid_pos(p, s)
        s[idx] = np.inf

    return s


def write_frame(p, fig):
    fig.canvas.draw()
    data = fig.canvas.tostring_rgb()
    width, height = fig.canvas.get_width_height()
    im = Image.frombytes("RGB", (width, height), data)
    im.save(p.stdin, "png")


if __name__ == "__main__":
    __args = tyro.cli(Args)
    main(__args)
