from dataclasses import dataclass
import os
import pickle
from subprocess import Popen, PIPE
import warnings
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import tqdm
import tyro
from common import ShapeType, EstimationResult, EstimationMethod, BaselineParams
import myenv
import estimate_oracle as oracle

std_f = 0.05
std_m = 0.005


@dataclass
class Args:
    outdir: str = "result/result_baseline.pickle"
    """Output directory"""
    load: str = ""
    """Filename of pre-recorded force signals if use"""
    plot: bool = True
    """Enable plotting"""
    shape: ShapeType = ShapeType.arch
    """Expected tool shape"""
    rho: float = 0.9
    """Forgetting factor of the recursive least squares"""
    alpha: float = 1e6
    """Initialization parameter"""
    save_animation: bool = False
    """Save animation of estimation"""
    progressbar: bool = True
    """Show a progress bar"""


def main(args: Args):
    if os.path.dirname(args.outdir) != "":
        os.makedirs(os.path.dirname(args.outdir), exist_ok=True)

    # Initialization
    if args.load != "":
        env = myenv.ContactEnvReal(args.load, shape_type=args.shape)
        shape_lim_x = (-0.2, 0.2)
        shape_lim_y = (-0.1, 0.3)

        def init_estimation():
            _c0 = np.random.uniform(-0.1, 0.1)
            _c1 = np.random.uniform(-0.1, 0.1)
            return np.stack([_c0, _c1], axis=0)

    else:
        env = myenv.ContactEnvSim(std_f=std_f, std_m=std_m, shape_type=args.shape)
        shape_lim_x = (-0.0, 0.4)
        shape_lim_y = (-0.2, 0.2)

        def init_estimation():
            _c0 = np.random.uniform(-0.1, 0.1)
            _c1 = np.random.uniform(0.0, 0.2)
            return np.stack([_c0, _c1], axis=0)

    x_dim = 2

    c = init_estimation()
    Q_inv = np.identity(x_dim) / args.alpha
    estimating = False

    if args.plot or args.save_animation:
        fscale = 0.03
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_aspect("equal")
        ax1.set_xlim(shape_lim_x)
        ax1.set_ylim(shape_lim_y)
        env.plot(ax1, color="black", lw=3, solid_capstyle="round", zorder=1)
        positions = ax1.scatter([c[0]], [c[1]], color="black", zorder=3)
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
                color="black",
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
                color="black",
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

    # Setup record
    params = BaselineParams(rho=args.rho)
    record = EstimationResult(
        method=EstimationMethod.baseline, shape_type=args.shape, hyper_params=params
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
    k = 0
    while not env.is_finished:
        # Get measurement values
        f, m = env.step()

        if env.is_finished:
            break

        if np.linalg.norm(f) < 0.5:
            estimating = False
        else:
            if not estimating:
                c = init_estimation()
                Q_inv = np.identity(x_dim) / args.alpha
                estimating = True

            c, Q_inv = estimate_rls(c, Q_inv, f, m, args.rho)

        # Record results
        record.pos_true.append(env.pc.copy())
        record.force_true.append(env.fc.copy())
        record.observations.append(np.concatenate([f, m]))
        record.pos_est.append(c)

        # Plot estimation
        if (args.plot or args.save_animation) and k % frame_skip == 0:
            c_show = c.copy()
            if not estimating:
                c_show = np.full_like(c, np.nan)
            positions.set_offsets(c_show)
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
    with open(args.outdir, "wb") as f:
        pickle.dump(record, f)

    if args.save_animation:
        ffmpeg_process.stdin.close()
        ffmpeg_process.wait()


def estimate_rls(c, Q_inv, f, m, rho):
    F = np.array([-f[1], f[0]])
    Q_inv = rho * Q_inv + np.einsum("i,j->ij", F, F)

    Q = np.linalg.inv(Q_inv)
    c = c - np.dot(Q, F) * (m + np.dot(F, c))

    return c, Q_inv


def write_frame(p, fig):
    fig.canvas.draw()
    data = fig.canvas.tostring_rgb()
    width, height = fig.canvas.get_width_height()
    im = Image.frombytes("RGB", (width, height), data)
    im.save(p.stdin, "png")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
