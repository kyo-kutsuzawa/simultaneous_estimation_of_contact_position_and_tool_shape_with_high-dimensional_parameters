import logging
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import tyro
from common import (
    EstimationMethod,
    EstimationResult,
    ShapeType,
    ProposedParams,
    BaselineParams,
    NaiveParams,
    OracleParams,
)
from common_show_result import cm, search_results
import myenv


def main(
    dirname: str,
    /,
    figname: str = "result/result_position_errors_comparison.pdf",
    sim: bool = False,
) -> None:
    """Plot a position-estimation result in time series.

    Args:
        dirname: Directory in which files are to be plotted.
        figname: Figure path to be saved.
        sim: True for a simulation data, false for an experimental data.
    """
    datalist = []
    datalist += search_results(dirname, ProposedParams(), shape=ShapeType.undefined)
    datalist += search_results(dirname, BaselineParams(), shape=ShapeType.undefined)
    datalist += search_results(dirname, NaiveParams(), shape=ShapeType.undefined)
    datalist += search_results(dirname, OracleParams(), shape=ShapeType.undefined)

    if sim:
        count_start = 1000
    else:
        count_start = int(20 / 0.02)

    # Plot the result
    fig = show_position_errors_comparison(datalist, count_start)

    # Save the figure
    fig.savefig(figname)

    plt.show()


def show_position_errors_comparison(
    datalist: list[EstimationResult],
    count_start: int,
) -> matplotlib.figure.Figure:
    # Create lists of shapes (no duplication)
    list_shapes = [
        ShapeType.straight,
        ShapeType.arch,
        ShapeType.angular,
        ShapeType.wavy,
        ShapeType.knife,
    ]

    # Create lists of methods (no duplication)
    list_methods = [
        EstimationMethod.oracle,
        EstimationMethod.proposed,
        EstimationMethod.naive,
        EstimationMethod.baseline,
    ]
    list_method_names = [m.name for m in list_methods]

    fig = plt.figure(figsize=(8.5 * cm, 7.5 * cm))
    fig.subplots_adjust(
        left=0.14, bottom=0.16, right=0.99, top=0.99, wspace=0.1, hspace=0.1
    )
    gs = fig.add_gridspec(2, len(list_shapes), height_ratios=(0.3, 0.7))

    errors_list = [[[] for _ in list_methods] for _ in list_shapes]

    for data in datalist:
        # Read shape type and method
        s = list_shapes.index(data.shape_type)
        m = list_methods.index(data.method)

        # Extract results
        pos_true = np.stack(data.pos_true)
        pos_est = np.stack(data.pos_est)[:, 0:2]

        # Calculate estimation errors of contact position [cm]
        e = np.linalg.norm(pos_est - pos_true, axis=1)
        e_mean = np.nanmean(e[count_start:])
        errors_list[s][m].append(e_mean * 100)

    for s, shape_type in enumerate(list_shapes):
        # Plot shapes
        ax1 = fig.add_subplot(gs[0, s])
        ax1.set_aspect("equal")
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_xlim(-0.0, 0.4)
        ax1.set_ylim(-0.2, 0.2)
        env = myenv.ContactEnvSim(shape_type=shape_type)
        env.plot(ax1, color="red", solid_capstyle="round")
        if s == 0:
            ax1.set_ylabel("shape")

        # Plot errors
        ax2 = fig.add_subplot(gs[1, s])
        ax2.set_ylim(0, 20)
        if s != 0:
            ax2.set_yticks([])
        else:
            ax2.set_ylabel("Estimation errors [cm]")
        flierprops = dict(marker="o", markersize=3)
        ax2.boxplot(errors_list[s], flierprops=flierprops)
        ax2.set_xticklabels(list_method_names, rotation=90)

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
