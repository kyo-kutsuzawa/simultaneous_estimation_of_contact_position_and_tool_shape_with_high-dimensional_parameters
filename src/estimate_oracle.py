from dataclasses import dataclass
import os
import pickle
import warnings
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import tqdm
import tyro
from common import ShapeType, EstimationResult, EstimationMethod, OracleParams
import myenv

std_f = 0.05
std_m = 0.005


@dataclass
class Args:
    outdir: str = "result/result_oracle.pickle"
    """Output directory"""
    load: str = ""
    """Filename of pre-recorded force signals if use"""
    plot: bool = True
    """Enable plotting"""
    shape: ShapeType = ShapeType.arch
    """Expected tool shape"""


def main(args: Args) -> None:
    # Initialization
    if args.load != "":
        env = myenv.ContactEnvReal(args.load, shape_type=args.shape)
        tool_axis = "y"
    else:
        env = myenv.ContactEnvSim(std_f=std_f, std_m=std_m, shape_type=args.shape)
        tool_axis = "x"

    p_est = np.zeros((2,))
    estimating = False

    if args.plot:
        fscale = 0.2
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 1, 1)
        ax1.set_aspect("equal")
        if args.load != "":
            ax1.set_xlim(-0.2, 0.2)
            ax1.set_ylim(-0.1, 0.3)
        else:
            ax1.set_xlim(0.0, 0.4)
            ax1.set_ylim(-0.2, 0.2)
        env.plot(ax1, color="black")
        positions = ax1.scatter([0], [0])
        (force,) = ax1.plot(
            (env.pc[0], env.pc[0] + env.fc[0] * fscale),
            (env.pc[1], env.pc[1] + env.fc[1] * fscale),
            color="C1",
        )
        txt = plt.text(0.05, 0.15, "t = ")

    # Setup record
    params = OracleParams()
    record = EstimationResult(
        method=EstimationMethod.oracle, shape_type=args.shape, hyper_params=params
    )

    # Estimation loop
    k = 0
    pbar = tqdm.tqdm(
        total=env.t_max,
        bar_format="{l_bar}{bar}| {n:.2f}/{total:.2f} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
    )
    warnings.simplefilter(
        "ignore", tqdm.TqdmWarning
    )  # Ignore warning when the progress-bar value exceeds 1
    while not env.is_finished:
        # Get measurement values
        f, m = env.step()

        if env.is_finished:
            break

        if np.linalg.norm(f) < 0.5:
            estimating = False
        else:
            if not estimating:
                estimating = True

            p_est = estimate(f, m, env, tool_axis)

        # Record results
        record.pos_true.append(env.pc.copy())
        record.force_true.append(env.fc.copy())
        record.observations.append(np.concatenate([f, m]))
        record.pos_est.append(p_est.copy())

        # Plot estimation
        if args.plot and k % 10 == 0:
            positions.set_offsets([p_est[0], p_est[1]])
            force.set_data(
                (env.pc[0], env.pc[0] + env.fc[0] * fscale),
                (env.pc[1], env.pc[1] + env.fc[1] * fscale),
            )
            txt.set_text("t = {:5.3f}".format(env.t))

            if not estimating:
                positions.set_offsets([np.nan, np.nan])

            plt.pause(0.01)

        k += 1
        pbar.update(env.dt)

    pbar.close()

    # Save the record
    if os.path.dirname(args.outdir) != "":
        os.makedirs(os.path.dirname(args.outdir), exist_ok=True)
    with open(args.outdir, "wb") as f:
        pickle.dump(record, f)


def estimate(force, moment, env, tool_axis="x"):
    # Compute contact-position candidates
    f_norm = np.dot(force, force)
    cx = lambda alpha: alpha * force[0] + force[1] * moment / f_norm
    cy = lambda alpha: alpha * force[1] - force[0] * moment / f_norm

    # Solve an optimization problem
    if tool_axis == "x":
        obj_func = lambda alpha: (cy(alpha) - env.shape_func(cx(alpha))) ** 2
    elif tool_axis == "y":
        obj_func = lambda alpha: (cx(alpha) - env.shape_func(cy(alpha))) ** 2

    # Compute the optimal alpha while its error is less than 10^{-10}
    value = 1.0
    while value > 1e-10:
        x0 = np.random.uniform(-1, 1)
        res = optimize.minimize(obj_func, x0, tol=1e-10)
        value = res.fun

    # Compute the estimated contact position
    p_estimate = np.array([float(cx(res.x).flatten()[0]), float(cy(res.x).flatten()[0])])

    return p_estimate


def estimate_plane(force, moment, height):
    f_estimate = force.copy()

    p_normal = np.array([force[1], -force[0]]) / np.dot(force, force) * moment
    p_parallel = (height - p_normal[1]) * force / np.linalg.norm(force)
    p_estimate = p_normal + p_parallel

    return p_estimate, f_estimate


if __name__ == "__main__":
    __args = tyro.cli(Args)
    main(__args)
