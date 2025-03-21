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
    figname: str = "result/result_position_errors_varying_fluctuations.pdf",
    sim: bool = False,
) -> None:
    """Plot position-estimation errors for varying magnitude of force fluctuation.

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
    fig = show_position_errors_varying_fluctuations(datalist, count_start)

    # Save the figure
    fig.savefig(figname)

    plt.show()


def show_position_errors_varying_fluctuations(
    datalist: list[EstimationResult], count_start: int
) -> matplotlib.figure.Figure:
    # Create a list of fluctuations (sorted, no duplication)
    list_fluctuations = [data.force_fluctuation_amp for data in datalist]
    list_fluctuations = sorted(list(set(list_fluctuations)))

    # Create a list of x-tick labels
    list_ticklabels = []
    for x in list_fluctuations:
        if x == 0:
            s = "$0$"
        else:
            s = "$\\frac{\\pi}{" + str(int(1 / (x / np.pi))) + "}$"

    list_ticklabels = []
    for x in list_fluctuations:
        if x == 0:
            s = "$0$"
        else:
            s = "$\\frac{\\pi}{" + str(int(1 / (x / np.pi))) + "}$"
        list_ticklabels.append(s)

    # Create a figure
    fig = plt.figure(figsize=(7.5 * cm, 4.5 * cm), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    # Calculate estimation errors
    errors_list = [[] for _ in list_fluctuations]
    for data in datalist:
        # Read force-fluctuation amplitude
        i = list_fluctuations.index(data.force_fluctuation_amp)

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
    ax.plot(list_fluctuations, error_mean, label="Mean")
    ax.fill_between(
        list_fluctuations,
        error_mean - error_std,
        error_mean + error_std,
        alpha=0.3,
        label="Standard deviation",
    )

    # Setup axis label and ticks
    ax.set_xlim(xmin=min(list_fluctuations), xmax=max(list_fluctuations))
    ax.set_ylim(ymin=0.0)
    ax.set_xticks(list_fluctuations, list_ticklabels)
    ax.set_xlabel("Fluctuation amplitude [rad]")
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
