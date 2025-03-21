import concurrent.futures
import common
import estimate_baseline, estimate_oracle

if __name__ == "__main__":
    n_workers = 8

    list_shapes = [
        common.ShapeType.straight,
        common.ShapeType.arch,
        common.ShapeType.angular,
        common.ShapeType.wavy,
        common.ShapeType.knife,
    ]

    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        for i in range(10):
            for shape in list_shapes:
                args_baseline = estimate_baseline.Args(
                    outdir=f"result/sim/result_{shape.name}_{i:02d}_baseline.pickle",
                    shape=shape,
                    rho=0.992,
                    alpha=860,
                    plot=False,
                )
                executor.submit(estimate_baseline.main, args_baseline)

                args_oracle = estimate_oracle.Args(
                    outdir=f"result/sim/result_{shape.name}_{i:02d}_oracle.pickle",
                    shape=shape,
                    plot=False,
                )
                executor.submit(estimate_oracle.main, args_oracle)
