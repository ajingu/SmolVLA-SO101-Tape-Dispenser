# SmolVLA Fine-Tuning

Minimal SO-ARM data collection and evaluation helpers for SmolVLA fine-tuning.

This repository starts by vendoring the proven SO-ARM/LeRobot command wrappers
from the previous imitation-learning project, then keeps SmolVLA training and
augmentation work separate from that baseline project.

## Minimum Smoke Flow

```powershell
uv sync
uv run svla list-ports
uv run svla cameras --scan
uv run svla cameras --preview
uv run svla teleop --leader-port COM4 --follower-port COM3 --fps 30
uv run svla record --leader-port COM4 --follower-port COM3 --repo-id local/smolvla-smoke --episodes 2
uv run svla dataset-info --repo-id local/smolvla-smoke
```

Copy `configs/camera_config.example.json` to `configs/camera_config.json` and
edit it for the actual wrist/upper camera indices before recording with cameras.
