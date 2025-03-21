import concurrent.futures
import json
import os
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
        for i in range(10):
            for shape in list_shapes:
                args = estimate_proposed.Args(**params_dict)
                args.outdir = f"result/sim/result_{shape.name}_{i:02d}.pickle"
                args.shape = shape
                args.plot = False

                executor.submit(estimate_proposed.main, args)
