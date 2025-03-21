import concurrent
import copy
from dataclasses import dataclass
import datetime
import glob
import itertools
import logging
import os
import pickle
import typing
import numpy as np
import optuna
import tyro
from common import ShapeType, EstimationResult, ProposedParams
import estimate_proposed
import myenv

outdir = ""
n_workers = 1
n_rollouts = 0
n_particles = 0
cell_size = 0.0


@dataclass
class Args:
    n_workers: int = 1
    """Number of processes"""
    n_trials: int = 300
    """Number of trials for each process"""
    n_rollouts: int = 5
    """Number of rollouts for each trial"""
    n_particles: int = 250
    """Number of particles"""
    cell_size: float = 0.005
    """Size of the cell [m]"""
    storage_name: str = ""
    """Name of database"""
    study_name: str = ""
    """Name of study"""


def objective(trial: optuna.Trial) -> float:
    args_template = estimate_proposed.Args(
        plot=False,
        progressbar=False,
        n_particles=n_particles,
        cell_size=cell_size,
        d_th=trial.suggest_float("d_th", 0.001, 0.01, log=True),
        theta_th=trial.suggest_float("theta_th", np.pi / 48, np.pi / 6),
        s_inc=trial.suggest_float("s_inc", 1e-6, 1e-1, log=True),
        s_dec=trial.suggest_float("s_dec", 1e-6, 1e-1, log=True),
        eff_th=trial.suggest_float("eff_th", 0.1, 0.99),
        std_x=trial.suggest_float("std_x", 1e-5, 1e-3, log=True),
        std_y=trial.suggest_float("std_y", 1e-5, 1e-3, log=True),
        encode=estimate_proposed.EncodeMethod.softmax,
        shape=ShapeType.angular,
    )

    # Setup a list of all combinations of rollout number and shape type
    filelist = glob.glob(os.path.join(os.path.dirname(__file__), "../data", "*.csv"))
    params = list(itertools.product(list(range(n_rollouts)), filelist))

    list_position_errors = []
    list_shape_errors = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i, filename in params:
            # Setup arguments
            args = copy.deepcopy(args_template)
            args.outdir = os.path.join(
                outdir, f"trial_{trial.number}_{i:0>2d}_{os.path.basename(filename)}.pickle"
            )
            args.load = filename

            # Estimation and evaluation
            futures.append(executor.submit(estimate_and_evaluate, args))

        for result in futures:
            list_position_errors.append(result.result()[0])
            list_shape_errors.append(result.result()[1])

    # Compute mean errors
    mean_position_error = float(np.mean(list_position_errors))
    mean_shape_error = float(np.mean(list_shape_errors))

    return mean_position_error, mean_shape_error


def estimate_and_evaluate(args: estimate_proposed.Args) -> typing.Tuple[float, float]:
    # Run estimation
    estimate_proposed.main(args)

    # Load the estimation result
    with open(args.outdir, "rb") as f:
        data: EstimationResult = pickle.load(f)

    # Compute estimation errors
    rmse_position = compute_position_error(data)
    rmse_shape = compute_shape_error(data, sim=False)

    return rmse_position, rmse_shape


def compute_position_error(data: EstimationResult, idx_start: int = 20 * 50) -> float:
    # Read position data
    pos_true = np.stack(data.pos_true, axis=0)
    pos_est = np.stack(data.pos_est, axis=0)
    # observations = np.stack(data.observations, axis=0)

    # Remove non-contact data
    # force = observations[:, :2]
    # idx = np.linalg.norm(force, axis=1) < 0.5
    # idx = idx | (np.linalg.norm(pos_true, axis=1) > 0.6)
    idx = np.isnan(np.linalg.norm(pos_true, axis=1))
    pos_est[idx, :] = np.nan

    # Compute RMSE in cm
    rmse = np.linalg.norm(pos_true - pos_est, axis=1) * 100
    rmse = np.nanmean(rmse[idx_start:])

    return float(rmse)


def compute_shape_error(data: EstimationResult, sim: bool = True) -> float:
    def get_cell_center(
        i: int,
        j: int,
        grid: np.ndarray,
        extent: typing.Tuple[float, float, float, float],
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

    # Read hyper-parameters
    params: ProposedParams = data.hyper_params

    # Compute shape candidate by taking argmax over x- or y-axis
    shape_est_binary = np.zeros_like(data.shape_est)
    if sim:
        idx = np.argmax(data.shape_est, axis=0)
        for i in range(shape_est_binary.shape[1]):
            shape_est_binary[idx[i], i] = 1.0
    else:
        idx = np.argmax(data.shape_est, axis=1)
        for i in range(shape_est_binary.shape[0]):
            shape_est_binary[i, idx[i]] = 1.0

    # Compute coordinates of the shape candidates
    centers = []
    for j in range(shape_est_binary.shape[0]):
        for k in range(shape_est_binary.shape[1]):
            if shape_est_binary[j, k] > 0.5:
                c = get_cell_center(j, k, shape_est_binary, params.shape_extent)
                centers.append(c)
    centers = np.stack(centers, axis=0)

    # Create an environment for the true shape
    if sim:
        env = myenv.ContactEnvSim(shape_type=data.shape_type)
    else:
        env = myenv.ContactEnvReal(shape_type=data.shape_type)

    # Compute the shape RMSE in cm
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
    rmse = np.sqrt(np.nanmean(errors_array**2)) * 100

    return float(rmse)


if __name__ == "__main__":
    # Setup logger
    logger = logging.getLogger(__name__)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.propagate = False

    # Parse command-line arguments
    __args = tyro.cli(Args)

    # Setup variables
    n_workers = __args.n_workers
    n_trials = __args.n_trials
    n_rollouts = __args.n_rollouts
    n_particles = __args.n_particles
    cell_size = __args.cell_size
    time_str = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    storage_name = (
        __args.storage_name
        if __args.storage_name != ""
        else "sqlite:///result/shape-estimation.sqlite3"
    )
    study_name = (
        __args.study_name if __args.study_name != "" else f"shape-estimation-{time_str}"
    )
    outdir = f"result/optuna_{time_str}"

    # Setup a study
    study = optuna.create_study(
        directions=[
            optuna.study.StudyDirection.MINIMIZE,
            optuna.study.StudyDirection.MINIMIZE,
        ],
        storage=storage_name,
        study_name=study_name,
    )

    # Setup a study
    study = optuna.create_study(
        directions=[
            optuna.study.StudyDirection.MINIMIZE,
            optuna.study.StudyDirection.MINIMIZE,
        ],
        storage=storage_name,
        study_name=study_name,
        load_if_exists=True,
    )
    logger.info(
        "You can connect to the database by running "
        + f"`optuna-dashboard {storage_name}`"
    )
    study.set_metric_names(["position error", "shape error"])

    # Run optimization
    study.optimize(objective, n_trials)

    # Print the best results
    logger.info(study.best_trials)
