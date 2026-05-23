# SmolVLA Fine-Tuning for SO-ARM101 / SO-101

Behavior-cloning fine-tuning of SmolVLA for a real SO-ARM101/SO-101
tape-dispenser pick-and-place task.

https://github.com/user-attachments/assets/7a06d129-9cae-4e7a-829c-34e86cd26347

https://github.com/user-attachments/assets/65094197-6f90-4377-aafe-b6e1a47881d4

This project builds on the SO-ARM workflow from [my previous tape-dispenser
project](https://github.com/ajingu/Imitation-Learning_Tape-Dispenser), but uses
LeRobot SmolVLA as the default policy. It is intended for small real-world
manipulation experiments where camera placement, language instructions,
demonstration datasets, and rollout datasets need to be kept organized.

## Overview

- Robot: SO-ARM101/SO-101 leader arm + follower arm
- Cameras: 2 external USB cameras (wrist + side)
- Policy: SmolVLA via LeRobot

## Data and Evaluation

I collected 20 teleoperated demonstrations for a single language task:

`Put the blue tape dispenser in the red box.`

The fine-tuned policy completed the task in 5/8 evaluation rollouts with small
object position and rotation changes. Under the same robot, cameras, object,
and prompt, the base SmolVLA checkpoint completed 0/8 rollouts.

This is a small real-robot fine-tuning experiment, not a robust benchmark. The
main failure mode was recovery after a missed grasp: once the object was pushed
into an unseen pose, the policy often failed to re-grasp it.

- [Full 8-rollout fine-tuned evaluation](artifacts/SmolVLA_fine-tuned_8-evals.mp4)

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

### 6. Fine-Tune SmolVLA

Fine-tune from the SmolVLA base checkpoint:

```powershell
uv run svla train `
  --repo-id local/smolvla-smoke `
  --steps 3000 `
  --batch-size 8
```

Checkpoints are written under `outputs/train/` by default. Use Weights & Biases
with `--wandb --wandb-project <project>` if you want training curves.

### 7. Roll Out SmolVLA

Run the base SmolVLA checkpoint without fine-tuning as a smoke test:

```powershell
uv run svla rollout `
  --follower-port COM3 `
  --repo-id local/eval_smolvla-base-smoke `
  --task "Pick up the object" `
  --episodes 1 `
  --episode-time-s 20 `
  --fps 30
```

The base checkpoint is not adapted to your exact camera setup, object, or task,
so treat this as a pipeline, safety, and zero-shot comparison check.

Roll out a fine-tuned checkpoint:

```powershell
uv run svla rollout `
  --leader-port COM4 `
  --follower-port COM3 `
  --policy-path outputs/train/smolvla_local_smolvla-smoke/checkpoints/last/pretrained_model `
  --repo-id local/eval_smolvla-smoke `
  --episodes 3 `
  --episode-time-s 20 `
  --reset-time-s 15 `
  --fps 30 `
  --overwrite
```

Evaluation datasets with repo names beginning with `eval_` are stored under
`outputs/datasets/eval/`.

## Notes

- Camera names and placement matter. Keep camera configs, task names, and
  dataset repo IDs stable across recording, training, and rollout.
- More diverse demonstrations, especially recovery after missed grasps, are the
  most direct next step for improving robustness.

## References

- [LeRobot SmolVLA documentation](https://huggingface.co/docs/lerobot/en/smolvla)
- [LeRobot SO-101 documentation](https://huggingface.co/docs/lerobot/en/so101)
- [LeRobot real-world robot guide](https://huggingface.co/docs/lerobot/main/il_robots)
