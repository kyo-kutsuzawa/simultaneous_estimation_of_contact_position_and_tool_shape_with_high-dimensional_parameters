import logging
import matplotlib.figure
from matplotlib.gridspec import GridSpecFromSubplotSpec
import matplotlib.pyplot as plt
import numpy as np
import tyro
from common import EstimationResult, ShapeType, ProposedParams
from common_show_result import cm, search_results
import myenv


def main(
    dirname: str,
    /,
    figname: str = "result/result_position_errors_varying_resolution.pdf",
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
    fig = show_position_errors_varying_resolutions(datalist, count_start)

    # Save the figure
    fig.savefig(figname)

    plt.show()


def show_position_errors_varying_resolutions(
    datalist: list[EstimationResult], count_start: int
) -> matplotlib.figure.Figure:
    # Create a list of cell sizes (sorted, no duplication)
    list_cell_sizes = []
    for data in datalist:
        params: ProposedParams = data.hyper_params
        cell_size = params.cell_size
        list_cell_sizes.append(cell_size)
    list_cell_sizes = sorted(list(set(list_cell_sizes)))

    # Create a list of x-tick labels
    list_ticklabels = []
    for r in list_cell_sizes:
        s = str(r * 100)
        list_ticklabels.append(s)

    # Create a figure
    fig = plt.figure(figsize=(7.5 * cm, 4.5 * cm), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    # Calculate estimation errors
    errors_list = [[] for _ in list_cell_sizes]
    for data in datalist:
        params: ProposedParams = data.hyper_params

        # Read the cell size
        i = list_cell_sizes.index(params.cell_size)

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
    ax.plot(list_cell_sizes, error_mean, label="Mean")
    ax.fill_between(
        list_cell_sizes,
        error_mean - error_std,
        error_mean + error_std,
        alpha=0.3,
        label="Standard deviation",
    )

    # Setup axis label and ticks
    ax.set_xlim(xmin=min(list_cell_sizes), xmax=max(list_cell_sizes))
    ax.set_ylim(ymin=0.0)
    ax.set_xticks(list_cell_sizes, list_ticklabels)
    ax.set_xlabel("Cell size [cm]")
    ax.set_ylabel("RMSE of contact position [cm]")
    ax.legend()

    return fig


def show_shapes_varying_resolutions(
    datalist: list[EstimationResult],
    sim: bool,
) -> matplotlib.figure.Figure:
    """
    Show estimated shapes for varying resolutions.
    """
    # Create lists of cell_size (sorted, no duplication)
    params_list: list[ProposedParams] = [data.hyper_params for data in datalist]
    list_cell_size = [params.cell_size for params in params_list]
    list_cell_size = sorted(list(set(list_cell_size)))

    # Create lists of shapes (no duplication)
    list_shapes = []
    for data in datalist:
        if data.shape_type != ShapeType.undefined:
            list_shapes.append(data.shape_type)
    list_shapes = list(set(list_shapes))

    # Create a figure
    fig = plt.figure(figsize=(8.5 * cm, 8 * cm))
    if sim:
        fig.subplots_adjust(
            left=0.1, bottom=0.08, right=0.94, top=0.96, wspace=0.05, hspace=0.1
        )
    gs = fig.add_gridspec(1, 2, width_ratios=(0.97, 0.03))

    # Create a gridspec of #shape x #cell_size
    n_row = len(list_shapes)
    n_col = len(list_cell_size)
    gs1 = GridSpecFromSubplotSpec(n_row, n_col, gs[0, 0])

    # Create axis labels
    ax = fig.add_subplot(gs[0, 0])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.text(
        -0.12,
        0.5,
        "$d_{\\mathrm{th}}$",
        va="center",
        transform=ax.transAxes,
        rotation=90,
    )
    ax.text(0.5, -0.12, "$\\theta_{\\mathrm{th}}$", ha="center", transform=ax.transAxes)

    plotted = np.full((n_row, n_col), False)
    for data in datalist:
        # Read d_th and theta_th
        params: ProposedParams = data.hyper_params
        s = list_shapes.index(data.shape_type)
        r = list_cell_size.index(params.cell_size)

        # Read the next data if the pair of parameters are already plotted
        if plotted[s, r]:
            continue

        # Setup an axis
        ax_s = fig.add_subplot(gs1[s, r])
        ax_s.set_aspect("equal")
        ax_s.set_xticks([])
        ax_s.set_yticks([])
        if s == 0:
            ax_s.set_title("${:.1f}$ cm".format(params.cell_size * 100), pad=-1)

        if sim:
            env = myenv.ContactEnvSim(shape_type=data.shape_type)
        else:
            env = myenv.ContactEnvReal(shape_type=data.shape_type)

        # Plot an estimated shape
        im = ax_s.imshow(
            data.shape_est,
            vmin=0.0,
            vmax=1.0,
            origin="lower",
            extent=params.shape_extent,
        )

        # Plot the ground-truth shape
        env.plot(ax_s, color="red", lw=1)

        # Add ticks and labels to the bottom left one
        if s == len(list_shapes) - 1 and r == 0:
            ax_s.set_xticks(params.shape_extent[0:2])
            ax_s.set_yticks(params.shape_extent[2:4])
            if sim:
                ax_s.set_xlabel("$x$ [m]", labelpad=-1)
                ax_s.set_ylabel("$y$ [m]", labelpad=-1)
            else:
                ax_s.set_xlabel("$y$ [m]", labelpad=0)
                ax_s.set_ylabel("$z$ [m]", labelpad=-2)

        plotted[s, r] = True

    # Show a color bar
    gs2 = GridSpecFromSubplotSpec(1, 1, gs[0, 1])
    ax_colorbar = fig.add_subplot(gs2[0, 0])
    plt.colorbar(im, cax=ax_colorbar, ticks=[0, 1])
    ax_colorbar.set_ylabel("Probability", labelpad=-3)

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
