from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from svla import cameras
from svla.commands import run

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "calibration"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "outputs" / "datasets"
DEFAULT_TRAIN_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "train"


def find_ports() -> int:
    return run(["lerobot-find-port"])


def find_ports_command(_: argparse.Namespace) -> int:
    return find_ports()


def calibration_dir(value: str | None = None) -> str:
    return value or str(DEFAULT_CALIBRATION_DIR)


def calibrate(
    role: str,
    port: str | None = None,
    robot_id: str | None = None,
    calibration_dir_value: str | None = None,
) -> int:
    robot_id = robot_id or role
    calib_dir = calibration_dir(calibration_dir_value)

    if role == "leader":
        return run(
            [
                "lerobot-calibrate",
                "--teleop.type=so101_leader",
                f"--teleop.port={port}",
                f"--teleop.id={robot_id}",
                f"--teleop.calibration_dir={calib_dir}",
            ]
        )

    return run(
        [
            "lerobot-calibrate",
            "--robot.type=so101_follower",
            f"--robot.port={port}",
            f"--robot.id={robot_id}",
            f"--robot.calibration_dir={calib_dir}",
        ]
    )


def calibrate_command(args: argparse.Namespace) -> int:
    return calibrate(
        args.role,
        port=args.port,
        robot_id=args.id,
        calibration_dir_value=args.calibration_dir,
    )


def teleoperate(
    leader_port: str | None = None,
    follower_port: str | None = None,
    leader_id: str | None = None,
    follower_id: str | None = None,
    calibration_dir_value: str | None = None,
    with_cameras: bool = False,
    camera_config: str | None = None,
    display_data: bool | None = None,
    fps: int = 60,
    max_relative_target: float | None = None,
    disable_torque_on_disconnect: bool = True,
) -> int:
    leader_id = leader_id or "leader"
    follower_id = follower_id or "follower"
    calib_dir = calibration_dir(calibration_dir_value)

    command = [
        "lerobot-teleoperate",
        "--teleop.type=so101_leader",
        f"--teleop.port={leader_port}",
        f"--teleop.id={leader_id}",
        f"--teleop.calibration_dir={calib_dir}",
        "--robot.type=so101_follower",
        f"--robot.port={follower_port}",
        f"--robot.id={follower_id}",
        f"--robot.calibration_dir={calib_dir}",
        f"--robot.disable_torque_on_disconnect={str(disable_torque_on_disconnect).lower()}",
        f"--fps={fps}",
    ]

    if max_relative_target is not None:
        command.append(f"--robot.max_relative_target={max_relative_target}")

    if with_cameras:
        camera_config_path = Path(camera_config or cameras.DEFAULT_CONFIG_PATH)
        command.append(f"--robot.cameras={cameras.lerobot_cameras_arg(camera_config_path)}")

    if display_data is None:
        display_data = with_cameras

    command.append(f"--display_data={str(display_data).lower()}")

    return run(command)


def teleoperate_command(args: argparse.Namespace) -> int:
    return teleoperate(
        leader_port=args.leader_port,
        follower_port=args.follower_port,
        leader_id=args.leader_id,
        follower_id=args.follower_id,
        calibration_dir_value=args.calibration_dir,
        with_cameras=args.with_cameras,
        camera_config=args.camera_config,
        display_data=args.display_data,
        fps=args.fps,
        max_relative_target=args.max_relative_target,
        disable_torque_on_disconnect=args.disable_torque_on_disconnect,
    )


def lerobot_dataset_path(repo_id: str) -> Path:
    return DEFAULT_DATASET_DIR / Path(repo_id)


def remove_existing_dataset(repo_id: str) -> None:
    dataset_path = lerobot_dataset_path(repo_id).resolve()
    dataset_root = DEFAULT_DATASET_DIR.resolve()

    if dataset_root not in dataset_path.parents:
        raise RuntimeError(f"Refusing to remove path outside project datasets: {dataset_path}")

    if dataset_path.exists():
        print(f"Removing existing local dataset: {dataset_path}")
        shutil.rmtree(dataset_path)


def record_dataset(
    leader_port: str,
    follower_port: str,
    repo_id: str,
    task: str,
    leader_id: str | None = None,
    follower_id: str | None = None,
    calibration_dir_value: str | None = None,
    camera_config: str | None = None,
    dataset_root: str | None = None,
    episodes: int = 2,
    episode_time_s: int = 30,
    reset_time_s: int = 15,
    display_data: bool = True,
    push_to_hub: bool = False,
    resume: bool = False,
    overwrite: bool = False,
    encoder_threads: int = 2,
    wait_start: bool = True,
) -> int:
    leader_id = leader_id or "leader"
    follower_id = follower_id or "follower"
    calib_dir = calibration_dir(calibration_dir_value)
    camera_config_path = Path(camera_config or cameras.DEFAULT_CONFIG_PATH)
    dataset_root_path = Path(dataset_root) if dataset_root else lerobot_dataset_path(repo_id)

    if overwrite and resume:
        raise RuntimeError("--overwrite and --resume cannot be used together")

    if overwrite:
        remove_existing_dataset(repo_id)

    print("Record controls:")
    print("  Right arrow: finish the current episode/reset early")
    print("  Left arrow : discard and rerecord the current episode")
    print("  Esc        : stop recording")

    if wait_start:
        os.environ["SO101_CAMERA_CONFIG"] = str(camera_config_path)
        command = [sys.executable, "-m", "svla.record_wait_start"]
    else:
        command = ["lerobot-record"]

    command.extend(
        [
            "--teleop.type=so101_leader",
            f"--teleop.port={leader_port}",
            f"--teleop.id={leader_id}",
            f"--teleop.calibration_dir={calib_dir}",
            "--robot.type=so101_follower",
            f"--robot.port={follower_port}",
            f"--robot.id={follower_id}",
            f"--robot.calibration_dir={calib_dir}",
            f"--robot.cameras={cameras.lerobot_cameras_arg(camera_config_path)}",
            f"--display_data={str(display_data).lower()}",
            f"--dataset.repo_id={repo_id}",
            f"--dataset.root={dataset_root_path}",
            f"--dataset.num_episodes={episodes}",
            f"--dataset.single_task={task}",
            f"--dataset.episode_time_s={episode_time_s}",
            f"--dataset.reset_time_s={reset_time_s}",
            f"--dataset.push_to_hub={str(push_to_hub).lower()}",
            f"--resume={str(resume).lower()}",
            "--dataset.streaming_encoding=true",
            f"--dataset.encoder_threads={encoder_threads}",
        ]
    )

    return run(command)


def record_command(args: argparse.Namespace) -> int:
    return record_dataset(
        leader_port=args.leader_port,
        follower_port=args.follower_port,
        repo_id=args.repo_id,
        task=args.task,
        leader_id=args.leader_id,
        follower_id=args.follower_id,
        calibration_dir_value=args.calibration_dir,
        camera_config=args.camera_config,
        dataset_root=args.dataset_root,
        episodes=args.episodes,
        episode_time_s=args.episode_time_s,
        reset_time_s=args.reset_time_s,
        display_data=args.display_data,
        push_to_hub=args.push_to_hub,
        resume=args.resume,
        overwrite=args.overwrite,
        encoder_threads=args.encoder_threads,
        wait_start=args.wait_start,
    )


def rollout_policy(
    follower_port: str,
    policy_path: str,
    repo_id: str,
    task: str,
    leader_port: str | None = None,
    leader_id: str | None = None,
    follower_id: str | None = None,
    calibration_dir_value: str | None = None,
    camera_config: str | None = None,
    dataset_root: str | None = None,
    episodes: int = 3,
    episode_time_s: int = 30,
    reset_time_s: int = 15,
    display_data: bool = True,
    overwrite: bool = False,
    encoder_threads: int = 2,
    device: str = "cuda",
    use_amp: bool = True,
    interpolation_multiplier: int = 1,
    wait_start: bool = True,
) -> int:
    leader_id = leader_id or "leader"
    follower_id = follower_id or "follower"
    repo_id = _policy_eval_repo_id(repo_id)
    calib_dir = calibration_dir(calibration_dir_value)
    camera_config_path = Path(camera_config or cameras.DEFAULT_CONFIG_PATH)
    dataset_root_path = Path(dataset_root) if dataset_root else lerobot_dataset_path(repo_id)

    if overwrite:
        remove_existing_dataset(repo_id)

    print("Rollout controls:")
    print("  Right arrow: finish the current episode/reset early")
    print("  Left arrow : discard and rerecord the current episode")
    print("  Esc        : stop rollout")

    if wait_start:
        os.environ["SO101_CAMERA_CONFIG"] = str(camera_config_path)
        command = [sys.executable, "-m", "svla.record_wait_start"]
    else:
        command = ["lerobot-record"]

    command.extend(
        [
            "--robot.type=so101_follower",
            f"--robot.port={follower_port}",
            f"--robot.id={follower_id}",
            f"--robot.calibration_dir={calib_dir}",
            f"--robot.cameras={cameras.lerobot_cameras_arg(camera_config_path)}",
            f"--policy.path={policy_path}",
            f"--policy.device={device}",
            f"--policy.use_amp={str(use_amp).lower()}",
            f"--display_data={str(display_data).lower()}",
            f"--dataset.repo_id={repo_id}",
            f"--dataset.root={dataset_root_path}",
            f"--dataset.num_episodes={episodes}",
            f"--dataset.single_task={task}",
            f"--dataset.episode_time_s={episode_time_s}",
            f"--dataset.reset_time_s={reset_time_s}",
            "--dataset.push_to_hub=false",
            "--dataset.streaming_encoding=true",
            f"--dataset.encoder_threads={encoder_threads}",
            f"--interpolation_multiplier={interpolation_multiplier}",
        ]
    )

    if leader_port:
        command.extend(
            [
                "--teleop.type=so101_leader",
                f"--teleop.port={leader_port}",
                f"--teleop.id={leader_id}",
                f"--teleop.calibration_dir={calib_dir}",
            ]
        )

    return run(command)


def rollout_command(args: argparse.Namespace) -> int:
    return rollout_policy(
        follower_port=args.follower_port,
        policy_path=args.policy_path,
        repo_id=args.repo_id,
        task=args.task,
        leader_port=args.leader_port,
        leader_id=args.leader_id,
        follower_id=args.follower_id,
        calibration_dir_value=args.calibration_dir,
        camera_config=args.camera_config,
        dataset_root=args.dataset_root,
        episodes=args.episodes,
        episode_time_s=args.episode_time_s,
        reset_time_s=args.reset_time_s,
        display_data=args.display_data,
        overwrite=args.overwrite,
        encoder_threads=args.encoder_threads,
        device=args.device,
        use_amp=args.use_amp,
        interpolation_multiplier=args.interpolation_multiplier,
        wait_start=args.wait_start,
    )


def dataset_path_command(args: argparse.Namespace) -> int:
    print(lerobot_dataset_path(args.repo_id))
    return 0


def dataset_open_command(args: argparse.Namespace) -> int:
    dataset_path = lerobot_dataset_path(args.repo_id)
    print(dataset_path)

    if not dataset_path.exists():
        print("Dataset path does not exist.")
        return 1

    if os.name == "nt":
        subprocess.Popen(["explorer", str(dataset_path)])
        return 0

    return run(["xdg-open", str(dataset_path)])


def dataset_info_command(args: argparse.Namespace) -> int:
    info_path = lerobot_dataset_path(args.repo_id) / "meta" / "info.json"
    if not info_path.exists():
        print(f"Dataset info not found: {info_path}")
        return 1

    info = json.loads(info_path.read_text(encoding="utf-8"))
    video_keys = [
        key for key, value in info.get("features", {}).items() if value.get("dtype") == "video"
    ]

    print(f"repo_id: {args.repo_id}")
    print(f"path: {lerobot_dataset_path(args.repo_id)}")
    print(f"total_episodes: {info.get('total_episodes')}")
    print(f"total_frames: {info.get('total_frames')}")
    print(f"fps: {info.get('fps')}")
    print(f"splits: {info.get('splits')}")
    print(f"video_keys: {', '.join(video_keys)}")
    return 0


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(":", "_")


def _policy_eval_repo_id(repo_id: str) -> str:
    namespace, separator, name = repo_id.rpartition("/")
    if name.startswith("eval_"):
        return repo_id

    eval_name = f"eval_{name}"
    eval_repo_id = f"{namespace}{separator}{eval_name}" if separator else eval_name
    print(
        "LeRobot requires policy rollout dataset names to start with 'eval_'. "
        f"Using repo_id={eval_repo_id}"
    )
    return eval_repo_id


def train_policy(
    repo_id: str,
    dataset_root: str | None = None,
    policy: str = "act",
    device: str = "cuda",
    output_dir: str | None = None,
    job_name: str | None = None,
    steps: int = 3000,
    batch_size: int = 8,
    save_freq: int = 1000,
    eval_freq: int = 0,
    use_amp: bool = True,
    wandb: bool = False,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_mode: str | None = None,
    push_to_hub: bool = False,
    policy_repo_id: str | None = None,
) -> int:
    dataset_root_path = Path(dataset_root) if dataset_root else lerobot_dataset_path(repo_id)
    run_name = job_name or f"{policy}_{_safe_name(repo_id)}"
    output_dir_path = Path(output_dir) if output_dir else DEFAULT_TRAIN_OUTPUT_DIR / run_name

    command = [
        sys.executable,
        "-m",
        "train",
        f"--dataset.repo_id={repo_id}",
        f"--dataset.root={dataset_root_path}",
        f"--policy.type={policy}",
        f"--policy.device={device}",
        f"--policy.use_amp={str(use_amp).lower()}",
        f"--policy.push_to_hub={str(push_to_hub).lower()}",
        f"--output_dir={output_dir_path}",
        f"--job_name={run_name}",
        f"--steps={steps}",
        f"--batch_size={batch_size}",
        f"--save_freq={save_freq}",
        f"--eval_freq={eval_freq}",
        f"--wandb.enable={str(wandb).lower()}",
        "--wandb.disable_artifact=true",
    ]

    if wandb_project:
        command.append(f"--wandb.project={wandb_project}")
    if wandb_entity:
        command.append(f"--wandb.entity={wandb_entity}")
    if wandb_mode:
        command.append(f"--wandb.mode={wandb_mode}")
    if policy_repo_id:
        command.append(f"--policy.repo_id={policy_repo_id}")

    return run(command)


def train_command(args: argparse.Namespace) -> int:
    return train_policy(
        repo_id=args.repo_id,
        dataset_root=args.dataset_root,
        policy=args.policy,
        device=args.device,
        output_dir=args.output_dir,
        job_name=args.job_name,
        steps=args.steps,
        batch_size=args.batch_size,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        use_amp=args.use_amp,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        push_to_hub=args.push_to_hub,
        policy_repo_id=args.policy_repo_id,
    )


def replay_episode(
    follower_port: str,
    repo_id: str,
    episode: int,
    follower_id: str | None = None,
    calibration_dir_value: str | None = None,
) -> int:
    follower_id = follower_id or "follower"
    calib_dir = calibration_dir(calibration_dir_value)

    return run(
        [
            "lerobot-replay",
            "--robot.type=so101_follower",
            f"--robot.port={follower_port}",
            f"--robot.id={follower_id}",
            f"--robot.calibration_dir={calib_dir}",
            f"--dataset.repo_id={repo_id}",
            f"--dataset.episode={episode}",
        ]
    )


def replay_command(args: argparse.Namespace) -> int:
    return replay_episode(
        follower_port=args.follower_port,
        repo_id=args.repo_id,
        episode=args.episode,
        follower_id=args.follower_id,
        calibration_dir_value=args.calibration_dir,
    )


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    find_parser = subparsers.add_parser("find-ports")
    find_parser.set_defaults(func=find_ports_command)

    calibrate_parser = subparsers.add_parser("calibrate")
    calibrate_parser.add_argument("role", choices=["leader", "follower"])
    calibrate_parser.add_argument("--port", required=True)
    calibrate_parser.add_argument("--id")
    calibrate_parser.add_argument("--calibration-dir")
    calibrate_parser.set_defaults(func=calibrate_command)

    teleop_parser = subparsers.add_parser("teleop")
    teleop_parser.add_argument("--leader-port", required=True)
    teleop_parser.add_argument("--follower-port", required=True)
    teleop_parser.add_argument("--leader-id")
    teleop_parser.add_argument("--follower-id")
    teleop_parser.add_argument("--calibration-dir")
    teleop_parser.add_argument("--with-cameras", action="store_true")
    teleop_parser.add_argument("--camera-config")
    teleop_parser.add_argument("--fps", type=int, default=60)
    teleop_parser.add_argument("--max-relative-target", type=float)
    teleop_parser.add_argument(
        "--no-disable-torque-on-disconnect",
        action="store_false",
        dest="disable_torque_on_disconnect",
    )
    display_group = teleop_parser.add_mutually_exclusive_group()
    display_group.add_argument("--display-data", action="store_true", dest="display_data")
    display_group.add_argument("--no-display-data", action="store_false", dest="display_data")
    teleop_parser.set_defaults(display_data=None)
    teleop_parser.set_defaults(disable_torque_on_disconnect=True)
    teleop_parser.set_defaults(func=teleoperate_command)

    record_parser = subparsers.add_parser("record")
    record_parser.add_argument("--leader-port", required=True)
    record_parser.add_argument("--follower-port", required=True)
    record_parser.add_argument("--repo-id", required=True)
    record_parser.add_argument(
        "--task",
        default="Pick up the object",
    )
    record_parser.add_argument("--leader-id")
    record_parser.add_argument("--follower-id")
    record_parser.add_argument("--calibration-dir")
    record_parser.add_argument("--camera-config")
    record_parser.add_argument("--dataset-root")
    record_parser.add_argument("--episodes", type=int, default=2)
    record_parser.add_argument("--episode-time-s", type=int, default=30)
    record_parser.add_argument("--reset-time-s", type=int, default=15)
    record_parser.add_argument("--encoder-threads", type=int, default=2)
    record_parser.add_argument("--push-to-hub", action="store_true")
    record_parser.add_argument("--resume", action="store_true")
    record_parser.add_argument("--overwrite", action="store_true")
    record_parser.add_argument("--no-wait-start", action="store_false", dest="wait_start")
    display_group = record_parser.add_mutually_exclusive_group()
    display_group.add_argument("--display-data", action="store_true", dest="display_data")
    display_group.add_argument("--no-display-data", action="store_false", dest="display_data")
    record_parser.set_defaults(display_data=True)
    record_parser.set_defaults(wait_start=True)
    record_parser.set_defaults(func=record_command)

    rollout_parser = subparsers.add_parser("rollout")
    rollout_parser.add_argument("--leader-port")
    rollout_parser.add_argument("--follower-port", required=True)
    rollout_parser.add_argument("--policy-path", required=True)
    rollout_parser.add_argument("--repo-id", default="local/eval_so101-rollout-smoke")
    rollout_parser.add_argument(
        "--task",
        default="Pick up the object",
    )
    rollout_parser.add_argument("--leader-id")
    rollout_parser.add_argument("--follower-id")
    rollout_parser.add_argument("--calibration-dir")
    rollout_parser.add_argument("--camera-config")
    rollout_parser.add_argument("--dataset-root")
    rollout_parser.add_argument("--episodes", type=int, default=3)
    rollout_parser.add_argument("--episode-time-s", type=int, default=30)
    rollout_parser.add_argument("--reset-time-s", type=int, default=15)
    rollout_parser.add_argument("--encoder-threads", type=int, default=2)
    rollout_parser.add_argument("--device", default="cuda")
    rollout_parser.add_argument("--no-use-amp", action="store_false", dest="use_amp")
    rollout_parser.add_argument("--interpolation-multiplier", type=int, default=1)
    rollout_parser.add_argument("--overwrite", action="store_true")
    rollout_parser.add_argument("--no-wait-start", action="store_false", dest="wait_start")
    display_group = rollout_parser.add_mutually_exclusive_group()
    display_group.add_argument("--display-data", action="store_true", dest="display_data")
    display_group.add_argument("--no-display-data", action="store_false", dest="display_data")
    rollout_parser.set_defaults(display_data=True)
    rollout_parser.set_defaults(use_amp=True)
    rollout_parser.set_defaults(wait_start=True)
    rollout_parser.set_defaults(func=rollout_command)

    dataset_path_parser = subparsers.add_parser("dataset-path")
    dataset_path_parser.add_argument("--repo-id", required=True)
    dataset_path_parser.set_defaults(func=dataset_path_command)

    dataset_open_parser = subparsers.add_parser("dataset-open")
    dataset_open_parser.add_argument("--repo-id", required=True)
    dataset_open_parser.set_defaults(func=dataset_open_command)

    dataset_info_parser = subparsers.add_parser("dataset-info")
    dataset_info_parser.add_argument("--repo-id", required=True)
    dataset_info_parser.set_defaults(func=dataset_info_command)

    replay_parser = subparsers.add_parser("replay")
    replay_parser.add_argument("--follower-port", required=True)
    replay_parser.add_argument("--repo-id", required=True)
    replay_parser.add_argument("--episode", type=int, default=0)
    replay_parser.add_argument("--follower-id")
    replay_parser.add_argument("--calibration-dir")
    replay_parser.set_defaults(func=replay_command)
