import concurrent.futures
import json
import os
import numpy as np
import common
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
        for i in range(5):
            for shape in list_shapes:
                for d_th in (0.001, 0.005, 0.01, 0.03):
                    for theta_param in (48, 24, 12, 6, 3):
                        args = estimate_proposed.Args(**params_dict)
                        args.outdir = f"result/sim_params/result_{shape.name}_{int(d_th*1000):02d}_{theta_param:02d}_{i:02d}.pickle"
                        args.d_th = d_th
                        args.theta_th = np.pi / theta_param
                        args.shape = shape
                        args.plot = False

                        executor.submit(estimate_proposed.main, args)
