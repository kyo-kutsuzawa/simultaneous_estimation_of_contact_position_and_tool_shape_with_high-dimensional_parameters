import logging
import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import tyro
from common import EstimationResult, ShapeType, ProposedParams
from common_show_result import cm, search_results
import myenv


def main(
    dirname: str,
    /,
    figname: str = "result/result_position_errors_varying_params.pdf",
    sim: bool = False,
) -> None:
    """Plot a heatmap of position-estimation errors in time series [cm].

    Each cell indicates a position estimation error [cm] for certain d_th and theta_th.
    The values count errors when the contact force fluctuating.

    Args:
        dirname: Directory in which files are to be plotted.
        figname: Figure path to be saved.
        sim: True for a simulation data, false for an experimental data.
    """
    datalist = search_results(dirname, ProposedParams(), shape=ShapeType.arch)

    # Plot the result
    fig = show_position_errors_varying_params(datalist)

    # Save the figure
    fig.savefig(figname)

    plt.show()


def show_position_errors_varying_params(
    datalist: list[EstimationResult],
) -> matplotlib.figure.Figure:
    # Create lists of d_th and theta_th (sorted, no duplication)
    params_list: list[ProposedParams] = [data.hyper_params for data in datalist]
    list_d_th = [params.d_th for params in params_list]
    list_theta_th = [
        int(np.pi / params.theta_th) for params in params_list
    ]  # For theta_th, the denominator is used; e.g. 24 if theta_th = pi / 24
    list_d_th = sorted(list(set(list_d_th)))
    list_theta_th = sorted(list(set(list_theta_th)))

    # Create a figure
    fig = plt.figure(figsize=(8.5 * cm, 6 * cm))
    fig.subplots_adjust(
        left=0.165, bottom=0.12, right=0.93, top=0.95, wspace=0.1, hspace=0.1
    )
    gs = fig.add_gridspec(1, 2, width_ratios=(0.95, 0.05))

    # Create axis ticks and labels
    ax_s = fig.add_subplot(gs[0, 0])
    ax_s.set_ylabel("$d_{\\mathrm{th}}$")
    ax_s.set_xlabel("$\\theta_{\\mathrm{th}}$")
    xticklabels = list(
        map(lambda x: "$\\frac{\\pi}{" + str(x) + "}$ rad", list_theta_th)
    )
    yticklabels = list(map(lambda x: "${}$ mm".format(int(x * 1000)), list_d_th))
    ax_s.set_xticks(list(range(len(xticklabels))), xticklabels)
    ax_s.set_yticks(list(range(len(yticklabels))), yticklabels)
    ax_s.tick_params(bottom=False, left=False, right=False, top=False)

    # Initialize errors list
    error_map = [[list() for _ in list_theta_th] for _ in list_d_th]

    for data in datalist:
        # Read d_th and theta_th
        params: ProposedParams = data.hyper_params
        x = list_d_th.index(params.d_th)
        y = list_theta_th.index(int(np.pi / params.theta_th))

        # Read position data
        pos_true = np.stack(data.pos_true, axis=0)
        pos_est = np.stack(data.pos_est, axis=0)
        observations = np.stack(data.observations, axis=0)

        # Remove non-contact data
        force = observations[:, :2]
        idx = np.linalg.norm(force, axis=1) < 0.5
        pos_est[idx, :] = np.nan

        # Compute RMSE in cm
        rmse = np.linalg.norm(pos_true - pos_est, axis=1) * 100
        error_map[x][y].append(rmse)

    # Compute the mean and std of RMSEs for each d_th and theta_th
    rmse_means = np.full((len(list_d_th), len(list_theta_th)), np.nan)
    rmse_stds = np.full((len(list_d_th), len(list_theta_th)), np.nan)
    for x in range(len(list_d_th)):
        for y in range(len(list_theta_th)):
            rmse_means[x, y] = np.nanmean(error_map[x][y])
            rmse_stds[x, y] = np.nanstd(error_map[x][y])

    # Plot a heatmap of position-estimation errors
    val_max = 20
    val_min = 0
    im = ax_s.imshow(rmse_means, vmin=val_min, vmax=val_max, cmap="cool")

    # Add a color bar
    ax_colorbar = fig.add_subplot(gs[0, 1])
    plt.colorbar(im, cax=ax_colorbar, ticks=[val_min, val_max])
    ax_colorbar.set_ylabel("Errors [cm]", labelpad=-5)

    # Add texts of error in the form of "mean +- std"
    for x in range(len(list_d_th)):
        for y in range(len(list_theta_th)):
            if rmse_means[x, y] < 10.0:
                text = "${:.2f}$\n$\\pm {:.2f}$".format(
                    rmse_means[x, y], rmse_stds[x, y]
                )
            else:
                text = "${:.1f}$\n$\\pm {:.2f}$".format(
                    rmse_means[x, y], rmse_stds[x, y]
                )
            ax_s.text(
                y,
                x,
                text,
                fontsize=8,
                horizontalalignment="center",
                verticalalignment="center",
            )

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
