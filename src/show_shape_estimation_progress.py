import logging
import matplotlib.figure
from matplotlib.gridspec import GridSpecFromSubplotSpec
import matplotlib.pyplot as plt
import tyro
from common import EstimationResult, ShapeType, ProposedParams, NaiveParams
from common_show_result import cm, search_results
import myenv


def main(
    dirname: str,
    /,
    figname: str = "result/result_shape_progress.pdf",
    sim: bool = False,
) -> None:
    """Plot shape-estimation progresses for different shapes.

    Args:
        dirname: Directory in which files are to be plotted.
        figname: Figure path to be saved.
        sim: True for a simulation data, false for an experimental data.
    """
    datalist = []
    datalist += search_results(dirname, ProposedParams(), shape=ShapeType.undefined)
    datalist += search_results(dirname, NaiveParams(), shape=ShapeType.undefined)

    if sim:
        list_counts = [0, 100, 200, 300, 500, 1000, 1500, -1]
    else:
        list_times = [0, 5, 10, 20, 30, 40]
        list_counts = [int(t / 0.02) for t in list_times]

    # Plot the result
    fig = show_shape_estimation_progress(datalist, list_counts, sim)

    # Save the figure
    fig.savefig(figname)

    plt.show()


def show_shape_estimation_progress(
    datalist: list[EstimationResult],
    list_counts: list[int],
    sim: bool,
) -> matplotlib.figure.Figure:
    # Create lists of shapes (no duplication)
    list_shapes = []
    for data in datalist:
        if len(data.shape_progress) > 0:
            if data.shape_type != ShapeType.undefined:
                list_shapes.append(data.shape_type)
    list_shapes = sorted(list(set(list_shapes)))
    shape_plotted = [False for _ in list_shapes]

    # Create a figure
    if sim:
        fig = plt.figure(figsize=(17.5 * cm, 12 * cm))
        fig.subplots_adjust(
            left=0.06, bottom=0.05, right=0.955, top=0.97, wspace=0.05, hspace=0.1
        )
    else:
        fig = plt.figure(figsize=(17.5 * cm, 10 * cm))
        fig.subplots_adjust(
            left=0.05, bottom=0.1, right=0.955, top=0.95, wspace=0.05, hspace=0.1
        )

    # Create a gridspec
    gs = fig.add_gridspec(1, 2, width_ratios=(0.985, 0.015))
    n_row = len(list_shapes)
    n_col = len(list_counts)
    gs1 = GridSpecFromSubplotSpec(n_row, n_col, gs[0, 0])

    for data in datalist:
        # Read shape type
        s = list_shapes.index(data.shape_type)

        if shape_plotted[s]:
            continue
        shape_plotted[s] = True

        if sim:
            env = myenv.ContactEnvSim(shape_type=data.shape_type)
        else:
            env = myenv.ContactEnvReal(shape_type=data.shape_type)

        # Plot shape-estimation progress over an episode
        for c, cnt in enumerate(list_counts):
            # Setup an axis
            ax_s = fig.add_subplot(gs1[s, c])
            ax_s.set_aspect("equal")
            ax_s.set_xticks([])
            ax_s.set_yticks([])
            if s == 0:
                if cnt == -1:
                    ax_s.set_title("$t={}$ s".format(int(len(data.shape_progress) * env.dt)), pad=-1)
                else:
                    ax_s.set_title("$t={}$ s".format(int(cnt * env.dt)), pad=-1)

            # Extract results
            params: ProposedParams | NaiveParams = data.hyper_params
            shape_est = data.shape_progress[cnt]

            # Plot estimated shape
            n_cells = int(0.4**2 / params.cell_size**2)
            vmax = 2.0 * 1.0 / (n_cells * params.cell_size**2)
            im = ax_s.imshow(
                shape_est,
                vmin=0.0,
                vmax=vmax,
                origin="lower",
                extent=params.shape_extent,
            )

            # Plot ground-truth shape
            env.plot(ax_s, color="red", lw=1)

            # Add ticks and axis labels
            if s == len(list_shapes) - 1 and c == 0:
                ax_s.set_xticks(params.shape_extent[0:2])
                ax_s.set_yticks(params.shape_extent[2:4])
                if sim:
                    ax_s.set_xlabel("$x$ [m]", labelpad=-1)
                    ax_s.set_ylabel("$y$ [m]", labelpad=-1)
                else:
                    ax_s.set_xlabel("$y$ [m]", labelpad=-3)
                    ax_s.set_ylabel("$z$ [m]", labelpad=-5)

            if not sim:
                ax_s.invert_xaxis()

    # Show a color bar
    ax_colorbar = fig.add_subplot(gs[0, 1])
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
