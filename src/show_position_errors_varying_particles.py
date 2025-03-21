import logging
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import tyro
from common import EstimationResult, ShapeType, ProposedParams
from common_show_result import cm, search_results


def main(
    dirname: str,
    /,
    figname: str = "result/result_position_errors_varying_particles.pdf",
    sim: bool = False,
) -> None:
    """Plot position-estimation errors for different numbers of particles.

    Args:
        dirname: Directory in which files are to be plotted.
        figname: Figure path to be saved.
        sim: True for a simulation data, false for an experimental data.
    """
    datalist = search_results(dirname, ProposedParams(), shape=ShapeType.undefined)

    if sim:
        count_start = 1000
    else:
        count_start = int(20 / 0.02)

    # Plot the result
    fig = show_position_errors_varying_particles(datalist, count_start)

    # Save the figure
    fig.savefig(figname)

    plt.show()


def show_position_errors_varying_particles(
    datalist: list[EstimationResult], count_start: int
) -> matplotlib.figure.Figure:
    # Create a list of particles (sorted, no duplication)
    list_n_particles = []
    for data in datalist:
        params: ProposedParams = data.hyper_params
        n_particles = params.n_particles
        list_n_particles.append(n_particles)
    list_n_particles = sorted(list(set(list_n_particles)))

    # Create a list of x-tick labels
    list_ticklabels = []
    for r in list_n_particles:
        s = str(r)
        list_ticklabels.append(s)

    # Create a figure
    fig = plt.figure(figsize=(7.5 * cm, 4.5 * cm), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    # Calculate estimation errors
    errors_list = [[] for _ in list_n_particles]
    for data in datalist:
        params: ProposedParams = data.hyper_params

        # Read the number of particles
        i = list_n_particles.index(params.n_particles)

        # Extract results
        pos_true = np.stack(data.pos_true)
        pos_est = np.stack(data.pos_est)[:, 0:2]

        # Calculate estimation errors of contact position [cm]
        e = np.linalg.norm(pos_est - pos_true, axis=1)
        e_mean = np.nanmean(e[count_start:])
        errors_list[i].append(e_mean * 100)

    # Calculate the mean and std. of errors
    error_mean = np.nanmean(errors_list, axis=1)
    error_std = np.nanstd(errors_list, axis=1)

    # plot errors
    ax.plot(list_n_particles, error_mean, label="Mean")
    ax.fill_between(
        list_n_particles,
        error_mean - error_std,
        error_mean + error_std,
        alpha=0.3,
        label="Standard deviation",
    )

    # Setup axis label and ticks
    ax.set_xlim(xmin=min(list_n_particles), xmax=max(list_n_particles))
    ax.set_ylim(ymin=0.0)
    ax.set_xscale("log")
    ax.set_xticks(list_n_particles, list_ticklabels)
    ax.set_xlabel("\\# Particles")
    ax.set_ylabel("RMSE of contact position [cm]")
    ax.legend()

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
