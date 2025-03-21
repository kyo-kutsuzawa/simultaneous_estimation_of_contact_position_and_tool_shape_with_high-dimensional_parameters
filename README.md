# Simultaneous estimation of contact position and tool shape with high-dimensional parameters using force measurements and particle filtering

## Search hyper-parameters

For simulation data:
```bash
python src/optimize_params_proposed.py
python src/optimize_params_baseline.py
python src/optimize_params_naive.py
```

For experimental data (assume measurement data are in `data`):
```bash
python src/optimize_params_proposed_exp.py
python src/optimize_params_baseline_exp.py
```

## Evaluation

For simulation data:
```bash
python src/evaluate_sim_proposed.py
python src/evaluate_sim_baseline.py
python src/evaluate_sim_naive.py
python src/evaluate_sim_delta.py
python src/evaluate_sim_fluctuations.py
python src/evaluate_sim_params.py
python src/evaluate_sim_particles.py
python src/evaluate_sim_progress.py
python src/evaluate_sim_resolutions.py
```

For experimental data (assume measurement data are in `data_{shape_name}`, e.g., `data_angular`):
```bash
python src/evaluate_exp.py
python src/evaluate_exp_progress.py
```

## Show evaluation results

For simulation data:
```bash
python src/show_shape_estimation_progress.py result/sim_progress --sim
python src/show_shape_estimation_progress.py result/sim_progress_naive --sim --figname result/result_shape_progress_naive.pdf
python src/show_position_errors_comparison.py result/sim --sim
python src/show_position_time_series.py result/sim/result_angular_00.pickle --sim --figname result/result_position_time_series_proposed.pdf
python src/show_position_time_series.py result/sim/result_angular_00_baseline.pickle --sim --figname result/result_position_time_series_baseline.pdf
python src/show_position_time_series.py result/sim/result_angular_00_naive.pickle --sim --figname result/result_position_time_series_naive.pdf
python src/show_shapes_varying_resolutions.py result/sim_resolution --sim
python src/show_position_errors_varying_resolutions.py result/sim_resolution --sim
python src/show_position_errors_varying_particles.py result/sim_particles --sim
python src/show_shapes_varying_params.py result/sim_params --sim
python src/show_position_errors_varying_params.py result/sim_params --sim
python src/show_shape_errors_comparison.py result/sim --sim
python src/show_shape_errors_varying_params.py result/sim_params --sim
python src/show_shapes_varying_s_delta.py result/sim_s_delta --sim
python src/show_position_errors_varying_fluctuations.py result/sim_fluctuation --sim
```

For experimental data:
```bash
python src/show_shape_estimation_progress.py result/exp_progress --figname result/result_shape_progress_exp.pdf
python src/show_shape_errors_comparison.py result/exp
python src/show_position_time_series.py result/exp/result_00_straight_record.pickle --figname result/result_position_time_series_exp_straight.pdf
python src/show_position_time_series.py result/exp/result_00_angular_record.pickle --figname result/result_position_time_series_exp_angular.pdf
python src/show_position_time_series.py result/exp/result_00_zigzag_record.pickle --figname result/result_position_time_series_exp_zigzag.pdf
python src/show_position_time_series.py result/exp/result_00_discontinuous_record.pickle --figname result/result_position_time_series_exp_discontinuous.pdf
python src/show_position_time_series.py result/exp/result_00_straight_record_baseline.pickle --figname result/result_position_time_series_exp_straight_baseline.pdf
python src/show_position_time_series.py result/exp/result_00_angular_record_baseline.pickle --figname result/result_position_time_series_exp_angular_baseline.pdf
python src/show_position_time_series.py result/exp/result_00_zigzag_record_baseline.pickle --figname result/result_position_time_series_exp_zigzag_baseline.pdf
python src/show_position_time_series.py result/exp/result_00_discontinuous_record_baseline.pickle --figname result/result_position_time_series_exp_discontinuous_baseline.pdf
```

## Generating animation videos

(assume measurement data are in `data_{shape_name}_for_video`, e.g., `data_angular_for_video`):
```bash
python src/record_animations.py
```
