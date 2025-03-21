import glob
import logging
import pickle
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import tyro
from common import EstimationResult
from common_show_result import cm


def main(
    filenames: str,
    /,
    figname: str = "result/result_position_time_series.pdf",
    sim: bool = False,
) -> None:
    """Plot a position-estimation result in time series.

    Args:
        filenames: File path to plot; if it contains "*", the mean and std. of multiple files will be plotted.
        figname: Figure path to be saved.
        sim: True for a simulation data, false for an experimental data.
    """
    if "*" in filenames:
        filelist = glob.glob(filenames)

        datalist = []
        for filename in filelist:
            with open(filename, "rb") as f:
                data: EstimationResult = pickle.load(f)
                datalist.append(data)

        # Plot the result
        fig = show_position_time_series_multiple(datalist, sim)
    else:
        with open(filenames, "rb") as f:
            data: EstimationResult = pickle.load(f)

        # Plot the result
        fig = show_position_time_series(data, sim)

    # Save the figure
    fig.savefig(figname)

    plt.show()


def show_position_time_series(
    data: EstimationResult,
    sim: bool,
) -> matplotlib.figure.Figure:
    if data is None:
        return

    pos_true = np.stack(data.pos_true, axis=0)
    pos_est = np.stack(data.pos_est, axis=0)
    observations = np.stack(data.observations, axis=0)

    idx = np.isnan(np.linalg.norm(pos_true, axis=1))
    pos_est[idx, :] = np.nan

    errors = np.linalg.norm(pos_true - pos_est, axis=1)

    # Create figures
    fig = plt.figure(figsize=(8.5 * cm, 8.5 * cm), constrained_layout=True)
    ax_p = fig.add_subplot(3, 1, 1)
    ax_e = fig.add_subplot(3, 1, 2)
    ax_f = fig.add_subplot(3, 1, 3)
    ax_m = ax_f.twinx()

    # Setup styles
    if sim:
        t_min = 0
        t_max = 20
        t_th = 10
        ax_p.set_ylim(-0.3, 1.5)
        ax_e.set_ylim(0.0, 0.3)
        ax_f.set_ylim(-2.75, 7.0)
        ax_m.set_ylim(0.0, 1.2)
        ax_p.set_xticks([0, 5, 10, 15, 20])
        ax_e.set_xticks([0, 5, 10, 15, 20])
        ax_f.set_xticks([0, 5, 10, 15, 20])
    else:
        t_min = 0
        t_max = len(data.pos_true) * 0.02
        t_th = 20
        ax_p.set_ylim(-0.1, 0.5)
        ax_e.set_ylim(0.0, 0.2)
        ax_f.set_ylim(-10.0, 25.0)
        ax_m.set_ylim(-1.0, 2.5)

    ax_p.set_xlim(t_min, t_max)
    ax_e.set_xlim(t_min, t_max)
    ax_f.set_xlim(t_min, t_max)
    ax_m.set_xlim(t_min, t_max)
    ax_p.set_ylabel("Contact position [m]")
    ax_e.set_ylabel("Estimation errors [m]")
    ax_f.set_ylabel("Measured force [N]")
    ax_m.set_ylabel("Measured moment [Nm]")
    ax_f.set_xlabel("Time [s]")

    # Shade before t_th
    ax_p.fill_between([0, t_th], [-10, -10], [10, 10], color="black", alpha=0.2)
    ax_e.fill_between([0, t_th], [-10, -10], [10, 10], color="black", alpha=0.2)
    ax_f.fill_between([0, t_th], [-30, -30], [30, 30], color="black", alpha=0.2)

    # Plot contact positions
    times = np.linspace(0, t_max, pos_true.shape[0])
    ax_p.plot(times, pos_true[:, 0], color="C0", ls="--", lw=0.5, label="True $c_x$")
    ax_p.plot(times, pos_true[:, 1], color="C1", ls="--", lw=0.5, label="True $c_y$")
    ax_p.plot(times, pos_est[:, 0], color="C0", lw=0.3, label="Estimated $c_x$")
    ax_p.plot(times, pos_est[:, 1], color="C1", lw=0.3, label="Estimated $c_y$")

    # Plot estimation errors
    times = np.linspace(0, t_max, pos_true.shape[0])
    ax_e.plot(times, errors, lw=0.3)

    # Plot observations
    ax_f.plot(times, observations[:, 0], color="C0", lw=0.3, label="$F_x$")
    ax_f.plot(times, observations[:, 1], color="C1", lw=0.3, label="$F_y$")
    ax_m.plot(times, observations[:, 2], color="C2", lw=0.3, label="$M$")

    # Add legends
    ax_p.legend(loc="upper left", ncol=2)
    ax_f.legend(loc="upper left", ncol=2)
    ax_m.legend(loc="upper right")

    return fig


def show_position_time_series_multiple(
    datalist: list[EstimationResult],
    sim: bool,
) -> matplotlib.figure.Figure:

    pos_est_list = []
    errors_list = []

    for data in datalist:
        pos_true = np.stack(data.pos_true, axis=0)
        pos_est = np.stack(data.pos_est, axis=0)
        observations = np.stack(data.observations, axis=0)

        idx = np.isnan(np.linalg.norm(pos_true, axis=1))
        pos_est[idx, :] = np.nan

        errors = np.linalg.norm(pos_true - pos_est, axis=1)

        pos_est_list.append(pos_est)
        errors_list.append(errors)

    pos_est_array = np.stack(pos_est_list, axis=0)
    pos_est_mean = np.nanmean(pos_est_array, axis=0)
    pos_est_std = np.nanstd(pos_est_array, axis=0)

    errors_array = np.stack(errors_list, axis=0)
    errors_mean = np.nanmean(errors_array, axis=0)
    errors_std = np.nanstd(errors_array, axis=0)

    # Create figures
    fig = plt.figure(figsize=(8.5 * cm, 8.5 * cm), constrained_layout=True)
    ax_p = fig.add_subplot(3, 1, 1)
    ax_e = fig.add_subplot(3, 1, 2)
    ax_f = fig.add_subplot(3, 1, 3)
    ax_m = ax_f.twinx()

    # Setup styles
    if sim:
        t_min = 0
        t_max = 20
        t_th = 10
        ax_p.set_ylim(-0.3, 1.5)
        ax_e.set_ylim(0.0, 0.3)
        ax_f.set_ylim(-2.75, 7.0)
        ax_m.set_ylim(0.0, 1.2)
        ax_p.set_xticks([0, 5, 10, 15, 20])
        ax_e.set_xticks([0, 5, 10, 15, 20])
        ax_f.set_xticks([0, 5, 10, 15, 20])
    else:
        t_min = 0
        t_max = len(data.pos_true) * 0.02
        t_th = 20
        ax_p.set_ylim(-0.1, 0.5)
        ax_e.set_ylim(0.0, 0.5)
        ax_f.set_ylim(-10.0, 25.0)
        ax_m.set_ylim(-1.0, 2.5)

    ax_p.set_xlim(t_min, t_max)
    ax_e.set_xlim(t_min, t_max)
    ax_f.set_xlim(t_min, t_max)
    ax_m.set_xlim(t_min, t_max)
    ax_p.set_ylabel("Contact position [m]")
    ax_e.set_ylabel("Estimation errors [m]")
    ax_f.set_ylabel("Measured force [N]")
    ax_m.set_ylabel("Measured moment [Nm]")
    ax_f.set_xlabel("Time [s]")

    # Shade before t_th
    ax_p.fill_between([0, t_th], [-10, -10], [10, 10], color="black", alpha=0.2)
    ax_e.fill_between([0, t_th], [-10, -10], [10, 10], color="black", alpha=0.2)
    ax_f.fill_between([0, t_th], [-30, -30], [30, 30], color="black", alpha=0.2)

    # Plot contact positions
    times = np.linspace(0, t_max, pos_true.shape[0])
    ax_p.plot(times, pos_true[:, 0], color="C0", ls="--", lw=0.5, label="True $c_x$")
    ax_p.plot(times, pos_true[:, 1], color="C1", ls="--", lw=0.5, label="True $c_y$")
    ax_p.plot(times, pos_est_mean[:, 0], color="C0", lw=0.3, label="Estimated $c_x$")
    ax_p.plot(times, pos_est_mean[:, 1], color="C1", lw=0.3, label="Estimated $c_y$")
    ax_p.fill_between(
        times,
        pos_est_mean[:, 0] - pos_est_std[:, 0],
        pos_est_mean[:, 0] + pos_est_std[:, 0],
        color="C0",
        alpha=0.3,
    )
    ax_p.fill_between(
        times,
        pos_est_mean[:, 1] - pos_est_std[:, 1],
        pos_est_mean[:, 1] + pos_est_std[:, 1],
        color="C1",
        alpha=0.3,
    )

    # Plot estimation errors
    times = np.linspace(0, t_max, pos_true.shape[0])
    ax_e.plot(times, errors_mean, lw=0.3, color="C0")
    ax_e.fill_between(
        times, errors_mean - errors_std, errors_mean + errors_std, color="C0", alpha=0.3
    )

    # Plot observations
    ax_f.plot(times, observations[:, 0], color="C0", lw=0.3, label="$F_x$")
    ax_f.plot(times, observations[:, 1], color="C1", lw=0.3, label="$F_y$")
    ax_m.plot(times, observations[:, 2], color="C2", lw=0.3, label="$M$")

    # Add legends
    ax_p.legend(loc="upper left", ncol=2)
    ax_f.legend(loc="upper left", ncol=2)
    ax_m.legend(loc="upper right")

    return fig


if __name__ == "__main__":
    # Setup logger
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    tyro.cli(main)
