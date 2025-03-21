import glob
import json
import os
import shutil
import common
import estimate_baseline
import estimate_proposed

if __name__ == "__main__":
    list_shapes_sim = [
        common.ShapeType.straight,
        common.ShapeType.arch,
        common.ShapeType.angular,
        common.ShapeType.wavy,
        common.ShapeType.knife,
    ]

    list_shapes_exp = [
        common.ShapeType.straight,
        common.ShapeType.angular,
        common.ShapeType.zigzag,
        common.ShapeType.discontinuous,
    ]

    params_dir = os.path.join(os.path.dirname(__file__), "optimized_params.json")
    with open(params_dir, "r") as f:
        optimized_params = json.load(f)
        params_proposed_sim_dict = optimized_params["proposed_sim"]
        params_baseline_sim_dict = optimized_params["baseline_sim"]
        params_proposed_exp_dict = optimized_params["proposed_exp"]
        params_baseline_exp_dict = optimized_params["baseline_exp"]

    for shape in list_shapes_sim:
        # Proposed @simulation
        args = estimate_proposed.Args(**params_proposed_sim_dict)
        args.outdir = f"result/anime/result_sim_{shape.name}.pickle"
        args.shape = shape
        args.plot = False
        args.save_animation = True
        estimate_proposed.main(args)
        shutil.copy2(
            "result/anime/video.mp4", f"result/anime/video_sim_{shape.name}.mp4"
        )

        # Proposed @simulation
        args = estimate_baseline.Args(**params_baseline_sim_dict)
        args.outdir = f"result/anime/result_sim_{shape.name}_baseline.pickle"
        args.shape = shape
        args.plot = False
        args.save_animation = True
        estimate_baseline.main(args)
        shutil.copy2(
            "result/anime/video.mp4",
            f"result/anime/video_sim_baseline_{shape.name}.mp4",
        )

    for shape in list_shapes_exp:
        filelist = glob.glob(
            os.path.join(os.path.dirname(__file__), f"../data_{shape.name}_for_video", "*.csv")
        )
        for filename in filelist:
            filename_body = os.path.splitext(os.path.basename(filename))[0]

            # Proposed @experiment
            args = estimate_proposed.Args(**params_proposed_exp_dict)
            args.outdir = (
                f"result/anime/result_exp_{shape.name}_{filename_body}.pickle"
            )
            args.load = filename
            args.shape = shape
            args.plot = False
            args.save_animation = True
            estimate_proposed.main(args)
            shutil.copy2(
                "result/anime/video.mp4",
                f"result/anime/video_exp_{shape.name}_{filename_body}.mp4",
            )

            # Baseline @experiment
            args = estimate_baseline.Args(**params_baseline_exp_dict)
            args.outdir = (
                f"result/anime/result_exp_{shape.name}_{filename_body}_baseline.pickle"
            )
            args.load = filename
            args.shape = shape
            args.plot = False
            args.save_animation = True
            estimate_baseline.main(args)
            shutil.copy2(
                "result/anime/video.mp4",
                f"result/anime/video_exp_baseline_{shape.name}_{filename_body}.mp4",
            )
