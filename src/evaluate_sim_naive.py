import concurrent.futures
import json
import os
import common
import estimate_naive

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
        params_dict = json.load(f)["naive_sim"]

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        for shape in list_shapes:
            for i in range(10):
                args = estimate_naive.Args(**params_dict)
                args.outdir = f"result/sim/result_{shape.name}_{i:02d}_naive.pickle"
                args.shape = shape
                args.plot = False

                executor.submit(estimate_naive.main, args)

            args = estimate_naive.Args(**params_dict)
            args.outdir = f"result/sim_progress_naive/result_{shape.name}.pickle"
            args.shape = shape
            args.save_all_shapes = True
            args.plot = False

            executor.submit(estimate_naive.main, args)
