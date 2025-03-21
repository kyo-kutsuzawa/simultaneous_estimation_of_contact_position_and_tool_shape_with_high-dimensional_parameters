import concurrent
from dataclasses import dataclass
import copy
import datetime
import glob
import itertools
import logging
import os
import pickle
import numpy as np
import optuna
import tyro
from common import ShapeType, EstimationResult
import estimate_baseline

outdir = ""
n_workers = 1
n_rollouts = 0


@dataclass
class Args:
    n_workers: int = 1
    """Number of processes"""
    n_trials: int = 300
    """Number of trials for each process"""
    n_rollouts: int = 5
    """Number of rollouts for each trial"""
    storage_name: str = ""
    """Name of database"""
    study_name: str = ""
    """Name of study"""


def objective(trial: optuna.Trial) -> float:
    args_template = estimate_baseline.Args(
        plot=False,
        progressbar=False,
        rho=trial.suggest_float("rho", 0.5, 1.0),
        alpha=trial.suggest_float("alpha", 1e0, 1e6, log=True),
        shape=ShapeType.angular,
    )

    # Setup a list of all combinations of rollout number and shape type
    shape_list = list(ShapeType)[1:]
    params = list(itertools.product(list(range(n_rollouts)), shape_list))

    list_position_errors1 = []
    list_position_errors2 = []

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = []
        for i, shape_type in params:
            # Setup arguments
            args = copy.deepcopy(args_template)
            args.outdir = os.path.join(
                outdir,
                f"trial_{trial.number}_{i:0>2d}_{shape_type.namepe}.pickle",
            )

            # Estimation and evaluation
            futures.append(executor.submit(estimate_and_evaluate, args))

        for result in futures:
            list_position_errors1.append(result.result()[0])
            list_position_errors2.append(result.result()[1])

    # Compute mean errors
    mean_position_error1 = float(np.mean(list_position_errors1))
    mean_position_error2 = float(np.mean(list_position_errors2))

    return mean_position_error1, mean_position_error2


def estimate_and_evaluate(args: estimate_baseline.Args) -> tuple[float, float]:
    # Run estimation
    estimate_baseline.main(args)

    # Load the estimation result
    with open(args.outdir, "rb") as f:
        data: EstimationResult = pickle.load(f)

    # Compute estimation errors
    idx_th = 1000
    rmse_position1 = compute_position_error(data, 0, idx_th)
    rmse_position2 = compute_position_error(data, idx_th, -1)

    return rmse_position1, rmse_position2


def compute_position_error(
    data: EstimationResult, idx_start: int, idx_end: int
) -> float:
    # Read position data
    pos_true = np.stack(data.pos_true, axis=0)
    pos_est = np.stack(data.pos_est, axis=0)
    # observations = np.stack(data.observations, axis=0)

    # Remove non-contact data
    # force = observations[:, :2]
    # idx = np.linalg.norm(force, axis=1) < 0.5
    idx = np.isnan(np.linalg.norm(pos_true, axis=1))
    pos_est[idx, :] = np.nan

    # Compute RMSE in cm
    rmse = np.linalg.norm(pos_true - pos_est, axis=1) * 100
    rmse = np.nanmean(rmse[idx_start:idx_end])

    return rmse


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
    study.set_metric_names(["error with fluctuations", "error without fluctuations"])

    # Run optimization
    study.optimize(objective, n_trials)

    # Print the best results
    logger.info(study.best_trials)
