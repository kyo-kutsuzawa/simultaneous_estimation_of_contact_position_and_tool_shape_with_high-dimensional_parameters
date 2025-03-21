import concurrent.futures
import glob
import json
import os
import common
import estimate_baseline
import estimate_oracle
import estimate_proposed

if __name__ == "__main__":
    n_workers = 16

    params_dir = os.path.join(os.path.dirname(__file__), "optimized_params.json")
    with open(params_dir, "r") as f:
        optimized_params = json.load(f)
        params_proposed_dict = optimized_params["proposed_exp"]
        params_baseline_dict = optimized_params["baseline_exp"]

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
                for i in range(5):
                    args = estimate_proposed.Args(**params_proposed_dict)
                    args.outdir = f"result/exp/result_{i:02d}_{shape_type.name}_{filename_body}.pickle"
                    args.load = filename
                    args.shape = shape_type
                    args.plot = False
                    executor.submit(estimate_proposed.main, args)

                    args_baseline = estimate_baseline.Args(**params_baseline_dict)
                    args_baseline.outdir = f"result/exp/result_{i:02d}_{shape_type.name}_{filename_body}_baseline.pickle"
                    args_baseline.load = filename
                    args_baseline.shape = shape_type
                    args_baseline.plot = False
                    executor.submit(estimate_baseline.main, args_baseline)

                args_oracle = estimate_oracle.Args(
                    outdir=f"result/exp/result_00_{shape_type.name}_{filename_body}_oracle.pickle",
                    load=filename,
                    shape=shape_type,
                    plot=False,
                )
                executor.submit(estimate_oracle.main, args_oracle)
