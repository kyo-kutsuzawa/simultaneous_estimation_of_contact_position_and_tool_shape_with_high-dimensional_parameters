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
        for shape in list_shapes:
            for n_particles in (10, 30, 100, 300, 1000):
                args = estimate_proposed.Args(**params_dict)
                args.outdir = (
                    f"result/sim_particles/result_{shape.name}_{n_particles:03d}.pickle"
                )
                args.n_particles = n_particles
                args.shape = shape
                args.plot = False

                executor.submit(estimate_proposed.main, args)
