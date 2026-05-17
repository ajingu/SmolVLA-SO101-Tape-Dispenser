# SmolVLA Fine-Tuning for SO-ARM101 / SO-101

SmolVLA data collection, augmentation, fine-tuning, and rollout helpers for a
real SO-ARM101/SO-101 leader/follower robot setup.

This project builds on the SO-ARM workflow from [my previous imitation-learning
project](https://github.com/ajingu/Imitation-Learning_Tape-Dispenser), but uses LeRobot SmolVLA as the default policy. It is intended for
small real-world manipulation experiments where camera placement, language
instructions, synthetic language-expanded datasets, and rollout datasets need
to be kept organized.

## Overview

- Robot: SO-ARM101/SO-101 leader arm + follower arm
- Cameras: external USB cameras configured through LeRobot OpenCV cameras
- Policy: SmolVLA via LeRobot
- Base checkpoint: `lerobot/smolvla_base`
- Optional synthetic data: language-expanded episode copies plus train-time
  image augmentation
- Training data: local LeRobot datasets under `outputs/datasets/train/`
- Evaluation data: policy rollouts under `outputs/datasets/eval/`

## Setup

Install dependencies:

```powershell
uv sync
```

Check that SmolVLA dependencies are available:

```powershell
uv run svla smolvla-check
```

The local `.env` file is ignored by Git. Add a Hugging Face token there if you
plan to push datasets or policies.

## Runbook

### 1. Find Robot Ports

```powershell
uv run svla list-ports
uv run svla find-ports
```

### 2. Calibrate Arms

```powershell
uv run svla calibrate leader --port COM4
uv run svla calibrate follower --port COM3
```

Use the same LeRobot IDs for calibration, teleoperation, data collection, and
rollout.

### 3. Check Teleoperation

```powershell
uv run svla teleop --leader-port COM4 --follower-port COM3 --fps 30
```

### 4. Configure Cameras

Scan camera indices:

```powershell
uv run svla cameras --scan
```

Create a local camera config:

```powershell
Copy-Item configs/cameras/camera_config.example.json configs/cameras/camera_config.json
```

Edit `configs/cameras/camera_config.json` for the real camera names and indices.
Then preview the configured cameras:

```powershell
uv run svla cameras --preview
```

Snapshots and preview captures are written to `outputs/camera_checks/`.

### 5. Record A Dataset

Record a short smoke dataset before collecting a longer run:

```powershell
uv run svla record `
  --leader-port COM4 `
  --follower-port COM3 `
  --repo-id local/smolvla-smoke `
  --task "Pick up the object" `
  --episodes 2
```

Recording controls: `Right arrow` finishes the current episode/reset early,
`Left arrow` rerecords the current episode, and `Esc` stops recording. By
default recording waits in a reset state before episode 0; press `Right arrow`
to start.

Inspect the dataset:

```powershell
uv run svla dataset-info --repo-id local/smolvla-smoke
uv run svla dataset-tasks --repo-id local/smolvla-smoke
uv run svla dataset-open --repo-id local/smolvla-smoke
```

Training datasets are stored under `outputs/datasets/train/`.

### 6. Optional Synthetic Data And Augmentation

SmolVLA conditions on both camera observations and language instructions. For a
small real-world dataset, it can be useful to test whether the policy benefits
from two lightweight augmentations:

- Language-expanded episodes: duplicate each demonstration once per task
  wording, assigning each copy a different `task_index`.
- Train-time image augmentation: randomly perturb brightness, contrast,
  saturation, hue, sharpness, and small affine transforms while training.

These augmentations are optional. They do not replace collecting more physical
variation, and they can be skipped for the first smoke train. The language
expansion is most useful when you want SmolVLA to see multiple natural-language
ways to describe the same behavior. Image augmentation is useful when lighting
or small camera differences are expected between training and rollout.

Task variants live in `configs/tasks/task_variants.example.json`:

```json
{
  "Pick up the cube and place it in the target area": [
    "Pick up the cube and place it in the target area.",
    "Move the cube into the target area.",
    "Grasp the cube and put it in the target area.",
    "Use the robot arm to place the cube in the target area."
  ]
}
```

Create a language-expanded dataset:

```powershell
uv run svla expand-language-episodes `
  --source-repo-id local/smolvla-smoke `
  --target-repo-id local/smolvla-smoke-lang-episodes `
  --variants-file configs/tasks/task_variants.example.json `
  --overwrite
```

Verify the expansion:

```powershell
uv run svla dataset-info --repo-id local/smolvla-smoke-lang-episodes
uv run svla dataset-tasks --repo-id local/smolvla-smoke-lang-episodes
```

`expand-language-episodes` duplicates the parquet rows for data and episode
metadata, assigns each copy a new `task_index`, and reuses the original video
files by timestamp range. It does not duplicate the mp4 files.

For train-time image augmentation, add `--image-aug` when training. The original
dataset files are not modified.

### 7. Fine-Tune SmolVLA

Fine-tune from the SmolVLA base checkpoint:

```powershell
uv run svla train `
  --repo-id local/smolvla-smoke `
  --steps 3000 `
  --batch-size 8
```

Train with language-expanded data and image augmentation:

```powershell
uv run svla train `
  --repo-id local/smolvla-smoke-lang-episodes `
  --steps 3000 `
  --batch-size 8 `
  --image-aug
```

Checkpoints are written under `outputs/train/` by default. Use Weights & Biases
with `--wandb --wandb-project <project>` if you want training curves.

### 8. Roll Out SmolVLA

Run the base SmolVLA checkpoint without fine-tuning as a smoke test:

```powershell
uv run svla rollout `
  --follower-port COM3 `
  --repo-id local/eval_smolvla-base-smoke `
  --task "Pick up the object" `
  --episodes 1 `
  --episode-time-s 30
```

The base checkpoint is not adapted to your exact camera setup, object, or task,
so treat this as a pipeline and safety check rather than a success benchmark.

Roll out a fine-tuned checkpoint:

```powershell
uv run svla rollout `
  --leader-port COM4 `
  --follower-port COM3 `
  --policy-path outputs/train/smolvla_local_smolvla-smoke/checkpoints/last/pretrained_model `
  --repo-id local/eval_smolvla-smoke `
  --episodes 3 `
  --episode-time-s 30 `
  --reset-time-s 15 `
  --overwrite
```

Evaluation datasets with repo names beginning with `eval_` are stored under
`outputs/datasets/eval/`.

## Notes

- Real-robot policies can move unpredictably. Keep the workspace clear and be
  ready to stop the process.
- Camera names and placement matter. Keep camera configs, task names, and
  dataset repo IDs stable across recording, training, and rollout.
- Language expansion is useful for testing the SmolVLA language path, but real
  task robustness still depends on physical variation in the demonstrations.

## References

- [LeRobot SmolVLA documentation](https://huggingface.co/docs/lerobot/en/smolvla)
- [LeRobot SO-101 documentation](https://huggingface.co/docs/lerobot/en/so101)
- [LeRobot real-world imitation learning guide](https://huggingface.co/docs/lerobot/main/il_robots)
