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
    figname: str = "result/result_shape_errors_varying_params.pdf",
    sim: bool = False,
) -> None:
    """Plot a heatmap of shape-estimation errors in time series [cm].

    Each cell indicates a shape-estimation error [cm] for certain d_th and theta_th.
    The values count errors when the contact force fluctuating.

    Args:
        dirname: Directory in which files are to be plotted.
        figname: Figure path to be saved.
        sim: True for a simulation data, false for an experimental data.
    """
    datalist = search_results(dirname, ProposedParams(), shape=ShapeType.arch)

    # Plot the result
    fig = show_shape_errors_varying_params(datalist, sim)

    # Save the figure
    fig.savefig(figname)

    plt.show()


def show_shape_errors_varying_params(
    datalist: list[EstimationResult],
    sim: bool,
) -> matplotlib.figure.Figure:
    def get_cell_center(
        i: int,
        j: int,
        grid: np.ndarray,
        extent: tuple[float, float, float, float],
    ) -> np.ndarray:
        """
        Compute the center position of the (i, j)-th cell
        """
        # Compute cell width and height
        n_row = grid.shape[0]
        n_col = grid.shape[1]
        cell_size_x = (extent[1] - extent[0]) / n_col
        cell_size_y = (extent[3] - extent[2]) / n_row

        # Compute upper-left corner position of the (i, j)-th cell
        corner_x = extent[0] + cell_size_x * j
        corner_y = extent[2] + cell_size_y * i

        # Compute center position of the (i, j)-th cell
        cx = corner_x + cell_size_x * 0.5
        cy = corner_y + cell_size_y * 0.5

        return np.array([cx, cy])

    # Create lists of d_th and theta_th (sorted, no duplication)
    params_list: list[ProposedParams] = [data.hyper_params for data in datalist]
    list_d_th = [params.d_th for params in params_list]
    list_theta_th = [
        int(np.pi / params.theta_th) for params in params_list
    ]  # For theta_th, the denominator is used ; e.g. 24 if theta_th = pi / 24
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

        # Compute shape candidate by taking argmax over x- or y-axis
        probability_th = 0.0
        shape_est_binary = np.zeros_like(data.shape_est)
        if sim:
            idx = np.argmax(data.shape_est, axis=0)
            for i in range(shape_est_binary.shape[1]):
                if data.shape_est[idx[i], i] > probability_th:
                    shape_est_binary[idx[i], i] = 1.0
        else:
            idx = np.argmax(data.shape_est, axis=1)
            for i in range(shape_est_binary.shape[0]):
                if data.shape_est[i, idx[i]] > probability_th:
                    shape_est_binary[i, idx[i]] = 1.0

        # Compute coordinates of the shape candidates
        centers = []
        for j in range(shape_est_binary.shape[0]):
            for k in range(shape_est_binary.shape[1]):
                if shape_est_binary[j, k] > 0.5:
                    c = get_cell_center(j, k, shape_est_binary, params.shape_extent)
                    centers.append(c)
        centers = np.stack(centers, axis=0)

        if sim:
            env = myenv.ContactEnvSim(shape_type=data.shape_type)
        else:
            env = myenv.ContactEnvReal(shape_type=data.shape_type)

        # Compute the shape estimation RMSE
        errors = []
        for c in centers:
            if sim:
                idx0, idx1 = 0, 1
            else:
                idx0, idx1 = 1, 0
            if env.x_range[0] <= c[idx0] <= env.x_range[1]:
                y_true = env.shape_func(c[idx0])
                e = c[idx1] - y_true
                errors.append(e)
        errors_array = np.stack(errors, axis=0)
        rmse = np.sqrt(np.mean(errors_array**2)) * 100
        error_map[x][y].append(rmse)

    # Compute the mean and std of RMSEs for each d_th and theta_th
    rmse_means = np.full((len(list_d_th), len(list_theta_th)), np.nan)
    rmse_stds = np.full((len(list_d_th), len(list_theta_th)), np.nan)
    for x in range(len(list_d_th)):
        for y in range(len(list_theta_th)):
            rmse_means[x, y] = np.nanmean(error_map[x][y])
            rmse_stds[x, y] = np.nanstd(error_map[x][y])

    # Plot a heatmap of shape-estimation errors
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
