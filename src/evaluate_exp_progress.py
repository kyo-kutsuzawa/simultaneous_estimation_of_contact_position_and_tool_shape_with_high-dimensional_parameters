import concurrent.futures
import glob
import json
import os
import common
import estimate_proposed

if __name__ == "__main__":
    n_workers = 16

    params_dir = os.path.join(os.path.dirname(__file__), "optimized_params.json")
    with open(params_dir, "r") as f:
        params_dict = json.load(f)["proposed_exp"]

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        for shape_type in (
            common.ShapeType.angular,
            common.ShapeType.straight,
            common.ShapeType.zigzag,
            common.ShapeType.discontinuous,
        ):
            filelist = glob.glob(
                os.path.join(
                    os.path.dirname(__file__), f"../data_{shape_type.name}", "*.csv"
                )
            )
            for filename in filelist:
                filename_body = os.path.splitext(os.path.basename(filename))[0]
                args = estimate_proposed.Args(**params_dict)
                args.outdir = f"result/exp_progress/result_{shape_type.name}_{filename_body}.pickle"
                args.load = filename
                args.shape = shape_type
                args.save_all_shapes = True
                args.plot = False
                executor.submit(estimate_proposed.main, args)
