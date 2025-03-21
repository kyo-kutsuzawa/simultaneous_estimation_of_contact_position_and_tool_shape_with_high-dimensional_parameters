import glob
import logging
import os
import pickle
import matplotlib.figure
import matplotlib.pyplot as plt
import tyro
from common import EstimationResult, ProposedParams, NaiveParams
from common_show_result import cm
import myenv


def main(
    dirname: str,
    /,
    outdir: str = "result",
    ext: str = "png",
    sim: bool = False,
    plot: bool = True,
) -> None:
    """Plot an estimated shape.

    Args:
        dirname: File path to plot or directory in which files are to be plotted.
        outdir: Directory to which figures are saved.
        ext: Extention of the figure (png, pdf, etc.).
        sim: True for a simulation data, false for an experimental data.
        plot: True to show figures.
    """
    filelist = []

    if os.path.isfile(dirname):
        filelist = [dirname]
    else:
        filelist = glob.glob(os.path.join(dirname, "*.pickle"))

    for filename in filelist:
        # Open a result
        with open(filename, "rb") as f:
            data: EstimationResult = pickle.load(f)

        # Plot the result
        fig = show_shape(data, sim)

        # Save the figure
        if fig is not None:
            filename_body = os.path.splitext(os.path.basename(filename))[0]
            figname = os.path.join(outdir, filename_body + "." + ext)
            fig.savefig(figname)

            if not plot:
                plt.close()

    if plot:
        plt.show()


def show_shape(
    data: EstimationResult,
    sim: bool,
) -> matplotlib.figure.Figure:
    if data is None:
        return

    if data.shape_est is None:
        return

    # Create a figure
    fig = plt.figure(figsize=(8.5 * cm, 7.5 * cm))
    if sim:
        fig.subplots_adjust(
            left=0.1, bottom=0.12, right=0.93, top=0.9, wspace=0.05, hspace=0.05
        )
    gs = fig.add_gridspec(1, 2, width_ratios=(0.97, 0.03))

    # Read d_th and theta_th
    params: ProposedParams | NaiveParams = data.hyper_params

    # Setup an axis
    ax_s = fig.add_subplot(gs[0, 0])
    ax_s.set_aspect("equal")
    ax_s.set_xticks([])
    ax_s.set_yticks([])

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
    ax_s.set_xlabel("$x$ [m]", labelpad=-3)
    ax_s.set_ylabel("$y$ [m]", labelpad=-10)
    ax_s.set_xticks(params.shape_extent[0:2])
    ax_s.set_yticks(params.shape_extent[2:4])

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
