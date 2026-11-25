# MapNet training analysis — Bebop v2_2

- Training runs included: **6**
- Runs with evaluation artifacts: **5**
- Excluded smoke tests: `bebop_mapnet_queue_smoke`

## Evaluation comparison

| Run | Split | Images used | Common images | Position mean (m) | RMSE (m) | P95 (m) | Orientation mean (°) | P95 (°) | Best val_loss | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bebop_mapnet_500 | val | 508 | 508 | 0.3571 | 0.4172 | 0.7831 | 9.9699 | 27.3341 | 0.547697 | 495 |
| bebop_mapnet_baseline | val | 508 | 508 | 0.4021 | 0.4760 | 0.9329 | 10.9574 | 24.4409 | 0.614461 | 277 |
| bebop_mapnet_lr5e-5 | val | 508 | 508 | 0.4234 | 0.5004 | 0.9433 | 11.3278 | 34.4832 | 0.660161 | 290 |
| bebop_mapnet_smoke | nan | — | 0 | — | — | — | — | — | 2.138078 | 1 |
| bebop_mapnet_wd1e-4 | val | 508 | 508 | 0.4006 | 0.4638 | 0.8714 | 11.3753 | 26.9230 | 0.604704 | 296 |
| bebop_mapnet_wd1e-4_2000 | val | 508 | 508 | 0.3477 | 0.4151 | 0.7985 | 8.3944 | 28.2830 | 0.554829 | 1311 |

## Common-image comparison

- Split `val`: **508** common images.

## Best runs on comparable images

- **Position mean:** bebop_mapnet_wd1e-4_2000 (0.3477 m)
- **Position RMSE:** bebop_mapnet_wd1e-4_2000 (0.4151 m)
- **Position P95:** bebop_mapnet_500 (0.7831 m)
- **Orientation mean:** bebop_mapnet_wd1e-4_2000 (8.3944 °)
- **Orientation P95:** bebop_mapnet_baseline (24.4409 °)

## Historical checkpoint evaluation

No historical checkpoints have been evaluated yet. Run with `--evaluate-checkpoints` to generate these results.

## Interpretation notes

- Training and validation losses are optimization losses, not geometric pose errors.
- Position and orientation errors are read from `eval.py` artifacts.
- Common-image statistics are calculated from the intersection of `valid_index` within each split.
- Validation and test splits are kept separate.
- RMSE penalizes large errors more strongly than the mean.
- P95 is the error value below which 95% of evaluated images fall.
- The 3D position error is the Euclidean distance between predicted and target XYZ positions.
- Trajectory plots connect poses in ascending `valid_index` order. They do not infer missing timestamps or sequence boundaries.
- The smoke-test run is excluded from this analysis.
