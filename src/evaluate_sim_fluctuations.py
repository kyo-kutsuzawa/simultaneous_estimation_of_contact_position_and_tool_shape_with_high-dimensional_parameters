import concurrent.futures
import json
import os
import common
import numpy as np
import estimate_proposed

if __name__ == "__main__":
    n_workers = 16

    list_shapes = [
        common.ShapeType.straight,
        common.ShapeType.arch,
        common.ShapeType.angular,
        common.ShapeType.wavy,
        common.ShapeType.knife,
    ]

    params_dir = os.path.join(os.path.dirname(__file__), "optimized_params.json")
    with open(params_dir, "r") as f:
        params_dict = json.load(f)["proposed_sim"]

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        for shape in list_shapes:
            for param in (0, 1 / 100, 1 / 48, 1 / 24, 1 / 12, 1 / 6):
                if param == 0:
                    param_str = "inf"
                else:
                    param_str = f"{int(1 / param):03d}"

                args = estimate_proposed.Args(**params_dict)
                args.outdir = (
                    f"result/sim_fluctuation/result_{shape.name}_{param_str}.pickle"
                )
                args.force_fluctuation_amp = np.pi * param
                args.shape = shape
                args.plot = False

                executor.submit(estimate_proposed.main, args)
