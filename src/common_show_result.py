from dataclasses import asdict
import glob
import logging
import os
import pickle
import traceback
import typing
import matplotlib.pyplot as plt
import common

# Setup for matplotlib
plt.rcParams["text.usetex"] = True
plt.rcParams["text.latex.preamble"] = r"\usepackage{times}"
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.size"] = 8
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["axes.titlesize"] = "medium"
cm = 1 / 2.54


def search_results(
    dirname: str,
    params: common.EstimationParams,
    shape: common.ShapeType = common.ShapeType.undefined,
) -> list[common.EstimationResult]:
    logger = logging.getLogger("search_results")

    # Get a list of pickle files in the directory
    filelist = glob.glob(os.path.join(dirname, "*.pickle"))
    filelist.sort()

    # Initialize a list of matched results
    datalist = []

    for filename in filelist:
        logger.debug(f"Check {filename}")

        # Load a result data
        with open(filename, "rb") as f:
            data: common.EstimationResult = pickle.load(f)

        # Check the method type
        if not isinstance(data.hyper_params, type(params)):
            continue

        # Check the shape
        if shape != common.ShapeType.undefined:
            if shape != data.shape_type:
                continue

        # Check parameter members
        found_diff = False
        for key in asdict(params).keys():
            if getattr(params, key) != None:
                if getattr(params, key) != getattr(data.hyper_params, key):
                    logger.debug(
                        f"Diff in {key}: expected = {getattr(params, key)}, actual = {getattr(data.hyper_params, key)}"
                    )
                    found_diff = True
                    break

        # Add the data if matched
        if not found_diff:
            datalist.append(data)
            logger.debug(f"Add {filename}")

    if len(datalist) == 0:
        logger.info("Nothing matched.")

    return datalist


def run_without_exit(
    func: typing.Callable, logger: logging.Logger, *args, **kwargs
) -> typing.Any:
    """Execute a function without exiting with errors (just printing error messages)."""
    try:
        return func(*args, **kwargs)
    except:
        traceback.print_exc()
        logger.error(f"Error occurred! {func.__name__}() halted.")
