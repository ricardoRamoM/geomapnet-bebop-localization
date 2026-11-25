# MapNet training analysis — Bebop v2_3

- Training runs included: **5**
- Runs with evaluation artifacts: **5**
- Excluded smoke tests: `bebop_mapnet_queue_smoke`

## Evaluation comparison

| Run | Split | Images used | Common images | Position mean (m) | RMSE (m) | P95 (m) | Orientation mean (°) | P95 (°) | Best val_loss | Best epoch |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bebop_mapnet_500 | val | 508 | 508 | 0.3322 | 0.4018 | 0.8283 | 10.9258 | 28.5743 | 0.622661 | 482 |
| bebop_mapnet_baseline | val | 508 | 508 | 0.3475 | 0.4150 | 0.8006 | 12.8885 | 45.7472 | 0.670173 | 290 |
| bebop_mapnet_lr5e-5 | val | 508 | 508 | 0.3769 | 0.4449 | 0.8332 | 12.7299 | 36.3677 | 0.728753 | 284 |
| bebop_mapnet_wd1e-4 | val | 508 | 508 | 0.3681 | 0.4326 | 0.8391 | 12.0160 | 34.3833 | 0.678797 | 259 |
| bebop_mapnet_wd1e-4_2000 | val | 508 | 508 | 0.3764 | 0.4468 | 0.8969 | 12.8091 | 43.8819 | 0.669946 | 232 |

## Common-image comparison

- Split `val`: **508** common images.

## Best runs on comparable images

- **Position mean:** bebop_mapnet_500 (0.3322 m)
- **Position RMSE:** bebop_mapnet_500 (0.4018 m)
- **Position P95:** bebop_mapnet_baseline (0.8006 m)
- **Orientation mean:** bebop_mapnet_500 (10.9258 °)
- **Orientation P95:** bebop_mapnet_500 (28.5743 °)

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
