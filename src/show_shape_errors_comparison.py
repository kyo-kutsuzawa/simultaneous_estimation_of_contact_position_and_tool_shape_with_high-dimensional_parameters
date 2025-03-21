import logging
import numpy as np
import tyro
from common import EstimationMethod, ShapeType, ProposedParams, NaiveParams
from common_show_result import search_results
import myenv


def main(
    dirname: str,
    /,
    sim: bool = False,
) -> None:
    """Compute and show shape-estimation errors [cm].

    Args:
        dirname: Directory in which files are to be plotted.
        sim: True for a simulation data, false for an experimental data.
    """
    datalist = []
    datalist += search_results(dirname, ProposedParams(), shape=ShapeType.undefined)
    datalist += search_results(dirname, NaiveParams(), shape=ShapeType.undefined)

    # Create lists of shapes (no duplication)
    list_shapes = [
        ShapeType.straight,
        ShapeType.arch,
        ShapeType.angular,
        ShapeType.wavy,
        ShapeType.knife,
        ShapeType.zigzag,
        ShapeType.discontinuous,
    ]
    list_shape_names = [s.name for s in list_shapes]
    max_name_length = max(map(len, list_shape_names))
    list_shape_names = list(map(lambda s: s.ljust(max_name_length), list_shape_names))

    # Create lists of methods (no duplication)
    list_methods = [
        EstimationMethod.proposed,
        EstimationMethod.naive,
    ]
    list_method_names = [m.name for m in list_methods]

    # Initialize errors list
    errors_list = [[[] for _ in list_methods] for _ in list_shapes]

    for data in datalist:
        # Read shape type and method
        params: ProposedParams = data.hyper_params
        s = list_shapes.index(data.shape_type)
        m = list_methods.index(data.method)

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

        # Compute the shape-estimation errors
        errors = []
        for c in centers:
            if sim:
                idx0, idx1 = 0, 1
            else:
                idx0, idx1 = 1, 0
            if env.x_range[0] <= c[idx0] <= env.x_range[1]:
                y_true = env.shape_func(c[idx0])
                e = abs(c[idx1] - y_true)
                errors.append(e)
        errors_array = np.stack(errors, axis=0)
        error = np.nanmean(errors_array) * 100
        errors_list[s][m].append(error)

    # Compute the mean and std of shape-estimation errors
    print("Shape".ljust(max_name_length), end=" & ")
    for m in list_method_names:
        print(m.capitalize().ljust(18), end=" ")
        if m == list_method_names[-1]:
            print("\\\\", end="")
        else:
            print("& ", end="")
    print()

    for x in range(len(list_shapes)):
        print(list_shape_names[x].capitalize(), end=" & ")
        for y in range(len(list_methods)):
            mean = np.nanmean(errors_list[x][y])
            std = np.nanstd(errors_list[x][y])
            print("${0:6.3f} \\pm {1:5.3f}$".format(mean, std), end=" ")
            if y == len(list_methods) - 1:
                print("\\\\", end="")
            else:
                print("& ", end="")
        print()


def get_cell_center(
    i: int, j: int, grid: np.ndarray, extent: tuple[float, float, float, float]
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


if __name__ == "__main__":
    # Setup logger
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    tyro.cli(main)
