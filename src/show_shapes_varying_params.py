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
    figname: str = "result/result_estimated_shapes_varying_params.pdf",
    sim: bool = False,
) -> None:
    """Plot estimated shapes for varying d_th and theta_th.

    Args:
        dirname: Directory in which files are to be plotted.
        figname: Figure path to be saved.
        sim: True for a simulation data, false for an experimental data.
    """
    datalist = search_results(dirname, ProposedParams(), shape=ShapeType.arch)

    # Plot the result
    fig = show_shapes_varying_params(datalist, sim)

    # Save the figure
    fig.savefig(figname)

    plt.show()


def show_shapes_varying_params(
    datalist: list[EstimationResult],
    sim: bool,
) -> matplotlib.figure.Figure:
    # Create lists of d_th and theta_th (sorted, no duplication)
    params_list: list[ProposedParams] = [data.hyper_params for data in datalist]
    list_d_th = [params.d_th for params in params_list]
    list_theta_th = [
        int(np.pi / params.theta_th) for params in params_list
    ]  # For theta_th, the denominator is used ; e.g. 24 if theta_th = pi / 24
    list_d_th = sorted(list(set(list_d_th)))
    list_theta_th = sorted(list(set(list_theta_th)))

    # Create a figure
    fig = plt.figure(figsize=(8.5 * cm, 6.7 * cm))
    if sim:
        fig.subplots_adjust(
            left=0.1, bottom=0.13, right=0.92, top=0.9, wspace=0.05, hspace=0.05
        )
    gs = fig.add_gridspec(1, 3, width_ratios=(0.95, 0.02, 0.03))

    # Create a gridspec of #d_th x #theta_th
    n_row = len(list_d_th)
    n_col = len(list_theta_th)
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
    ax.text(0.5, -0.15, "$\\theta_{\\mathrm{th}}$", ha="center", transform=ax.transAxes)

    plotted = np.full((n_row, n_col), False)
    for data in datalist:
        # Read d_th and theta_th
        params: ProposedParams = data.hyper_params
        x = list_d_th.index(params.d_th)
        y = list_theta_th.index(int(np.pi / params.theta_th))

        # Read the next data if the pair of parameters are already plotted
        if plotted[x, y]:
            continue

        # Setup an axis
        ax_s = fig.add_subplot(gs1[x, y])
        ax_s.set_aspect("equal")
        ax_s.set_xticks([])
        ax_s.set_yticks([])
        if x == len(list_d_th) - 1:
            ax_s.set_xlabel("$\\frac{\\pi}{" + str(list_theta_th[y]) + "}$ rad")
        if y == 0:
            if list_d_th[x] >= 0.01:
                ax_s.set_ylabel("${}$ cm".format(int(list_d_th[x] * 100)))
            else:
                ax_s.set_ylabel("${:.1f}$ cm".format(list_d_th[x] * 100))

        if sim:
            env = myenv.ContactEnvSim(shape_type=data.shape_type)
        else:
            env = myenv.ContactEnvReal(shape_type=data.shape_type)

        # Plot an estimated shape
        n_cells = int(0.4**2 / params.cell_size**2)
        vmax = 2.0 * 1.0 / (n_cells * params.cell_size**2)
        im = ax_s.imshow(
            data.shape_est,
            vmin=0.0,
            vmax=vmax,
            origin="lower",
            extent=params.shape_extent,
        )

        # Plot the ground-truth shape
        env.plot(ax_s, color="red", lw=1)

        # Add ticks and labels to the upper right one
        if x == 0 and y == len(list_theta_th) - 1:
            ax_s.set_xlabel("$x$ [m]", labelpad=-1)
            ax_s.set_ylabel("$y$ [m]", labelpad=-3)
            ax_s.set_xticks(params.shape_extent[0:2])
            ax_s.set_yticks(params.shape_extent[2:4])
            ax_s.xaxis.tick_top()
            ax_s.xaxis.set_label_position("top")
            ax_s.yaxis.tick_right()
            ax_s.yaxis.set_label_position("right")

        plotted[x, y] = True

    # Show a color bar
    gs2 = GridSpecFromSubplotSpec(2, 1, gs[0, 2], height_ratios=(0.3, 0.7))
    ax_colorbar = fig.add_subplot(gs2[1, 0])
    plt.colorbar(im, cax=ax_colorbar, ticks=[0, vmax])
    ax_colorbar.set_ylabel("Cell value", labelpad=-10)

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
