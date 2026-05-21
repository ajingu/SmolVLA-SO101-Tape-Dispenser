from __future__ import annotations

from pathlib import Path


def update_last_checkpoint_without_required_symlink(checkpoint_dir: Path) -> None:
    from lerobot.utils.constants import LAST_CHECKPOINT_LINK

    last_checkpoint_dir = checkpoint_dir.parent / LAST_CHECKPOINT_LINK
    if last_checkpoint_dir.is_symlink() or last_checkpoint_dir.is_file():
        last_checkpoint_dir.unlink()

    relative_target = checkpoint_dir.relative_to(checkpoint_dir.parent)
    try:
        last_checkpoint_dir.symlink_to(relative_target, target_is_directory=True)
    except OSError:
        marker_path = checkpoint_dir.parent / "last_checkpoint.txt"
        marker_path.write_text(str(relative_target), encoding="utf-8")


def main() -> None:
    from lerobot.scripts import lerobot_train
    from lerobot.utils import train_utils

    train_utils.update_last_checkpoint = update_last_checkpoint_without_required_symlink
    lerobot_train.update_last_checkpoint = update_last_checkpoint_without_required_symlink
    lerobot_train.main()


if __name__ == "__main__":
    main()
