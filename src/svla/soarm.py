from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from svla import cameras
from svla.commands import run

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "calibration"
DEFAULT_DATASET_DIR = PROJECT_ROOT / "outputs" / "datasets"
DEFAULT_TRAIN_DATASET_DIR = DEFAULT_DATASET_DIR / "train"
DEFAULT_EVAL_DATASET_DIR = DEFAULT_DATASET_DIR / "eval"
DEFAULT_TRAIN_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "train"
DEFAULT_SMOLVLA_POLICY_PATH = "lerobot/smolvla_base"
DEFAULT_SMOLVLA_ROLLOUT_REPO_ID = "local/eval_smolvla-base-rollout"


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


def relax_follower(
    follower_port: str,
    follower_id: str | None = None,
    calibration_dir_value: str | None = None,
) -> int:
    from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig
    from lerobot.robots.so_follower.so_follower import SOFollower

    follower_id = follower_id or "follower"
    calib_dir = calibration_dir(calibration_dir_value)
    config = SOFollowerRobotConfig(
        port=follower_port,
        id=follower_id,
        calibration_dir=Path(calib_dir),
        disable_torque_on_disconnect=True,
        cameras={},
    )
    robot = SOFollower(config)

    print("Support the follower arm before releasing torque.")
    robot.bus.connect()
    try:
        robot.bus.disable_torque(num_retry=5)
        print("Follower torque disabled.")
    finally:
        if robot.bus.is_connected:
            robot.bus.disconnect(disable_torque=True)

    return 0


def relax_command(args: argparse.Namespace) -> int:
    return relax_follower(
        follower_port=args.follower_port,
        follower_id=args.follower_id,
        calibration_dir_value=args.calibration_dir,
    )


def motor_scan(port: str, retries: int = 5) -> int:
    from lerobot.motors.feetech import FeetechMotorsBus

    bus = FeetechMotorsBus(port, {})
    bus.connect(handshake=False)
    try:
        motors = bus.broadcast_ping(num_retry=retries, raise_on_error=False) or {}
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)

    if not motors:
        print(f"No Feetech motors found on {port}.")
        return 1

    print(f"Feetech motors found on {port}:")
    for motor_id, model_number in sorted(motors.items()):
        print(f"  id={motor_id}: model={model_number}")

    expected = set(range(1, 7))
    found = set(motors)
    missing = sorted(expected - found)
    extra = sorted(found - expected)
    if missing:
        print(f"Missing expected SO101 ids: {missing}")
    if extra:
        print(f"Extra ids: {extra}")

    return 0 if not missing else 1


def motor_scan_command(args: argparse.Namespace) -> int:
    return motor_scan(args.port, retries=args.retries)


def motor_read_test(port: str, seconds: float = 10.0, hz: float = 20.0, retries: int = 5) -> int:
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    bus = FeetechMotorsBus(
        port=port,
        motors={
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        },
    )

    interval_s = 1.0 / hz
    deadline = time.perf_counter() + seconds
    reads = 0
    failures = 0
    latencies_ms: list[float] = []
    last_values = None

    bus.connect(handshake=True)
    try:
        while time.perf_counter() < deadline:
            start = time.perf_counter()
            try:
                last_values = bus.sync_read(
                    "Present_Position",
                    normalize=False,
                    num_retry=retries,
                )
                reads += 1
                latencies_ms.append((time.perf_counter() - start) * 1000)
            except Exception as exc:
                failures += 1
                print(f"read failure #{failures}: {exc}")

            sleep_s = interval_s - (time.perf_counter() - start)
            if sleep_s > 0:
                time.sleep(sleep_s)
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)

    print(f"reads={reads}, failures={failures}, seconds={seconds:.1f}, target_hz={hz:.1f}")
    if latencies_ms:
        print(
            "latency_ms="
            f"min={min(latencies_ms):.1f}, "
            f"mean={sum(latencies_ms) / len(latencies_ms):.1f}, "
            f"max={max(latencies_ms):.1f}"
        )
    if last_values is not None:
        print("last_positions_raw:")
        for motor, value in last_values.items():
            print(f"  {motor}: {value}")

    return 0 if failures == 0 else 1


def motor_read_test_command(args: argparse.Namespace) -> int:
    return motor_read_test(
        args.port,
        seconds=args.seconds,
        hz=args.hz,
        retries=args.retries,
    )


def motor_health(port: str, retries: int = 5, seconds: float = 0.0, hz: float = 10.0) -> int:
    from lerobot.motors import Motor, MotorNormMode
    from lerobot.motors.feetech import FeetechMotorsBus

    bus = FeetechMotorsBus(
        port=port,
        motors={
            "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
            "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
            "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
            "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
            "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
            "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
        },
    )

    fields = [
        "Present_Voltage",
        "Min_Voltage_Limit",
        "Max_Voltage_Limit",
        "Present_Temperature",
        "Status",
        "Present_Current",
        "Present_Load",
    ]

    def read_health_once() -> dict[str, dict[str, float | int | str]]:
        rows: dict[str, dict[str, float | int | str]] = {}
        for field in fields:
            try:
                values = bus.sync_read(field, normalize=False, num_retry=retries)
            except Exception as exc:
                print(f"failed to read {field}: {exc}")
                values = {}

            for motor, value in values.items():
                rows.setdefault(motor, {})[field] = value
        return rows

    rows: dict[str, dict[str, float | int | str]]
    samples: dict[str, list[int]] = {}
    status_samples: dict[str, list[int]] = {}

    bus.connect(handshake=True)
    try:
        if seconds > 0:
            interval_s = 1.0 / hz
            deadline = time.perf_counter() + seconds
            while time.perf_counter() < deadline:
                start = time.perf_counter()
                rows = read_health_once()
                for motor, data in rows.items():
                    voltage = data.get("Present_Voltage")
                    status = data.get("Status")
                    if isinstance(voltage, int):
                        samples.setdefault(motor, []).append(voltage)
                    if isinstance(status, int):
                        status_samples.setdefault(motor, []).append(status)

                sleep_s = interval_s - (time.perf_counter() - start)
                if sleep_s > 0:
                    time.sleep(sleep_s)

            rows = read_health_once()
        else:
            rows = read_health_once()
    finally:
        if bus.is_connected:
            bus.disconnect(disable_torque=False)

    print(f"Feetech motor health on {port}:")
    for motor, data in rows.items():
        present_v = data.get("Present_Voltage")
        min_v = data.get("Min_Voltage_Limit")
        max_v = data.get("Max_Voltage_Limit")
        voltage_text = "n/a"
        if isinstance(present_v, int | float):
            voltage_text = f"{present_v / 10:.1f} V raw={present_v}"

        limit_text = "n/a"
        if isinstance(min_v, int | float) and isinstance(max_v, int | float):
            limit_text = f"{min_v / 10:.1f}-{max_v / 10:.1f} V"

        print(
            f"  {motor}: voltage={voltage_text}, limits={limit_text}, "
            f"temp={data.get('Present_Temperature', 'n/a')}, "
            f"status={data.get('Status', 'n/a')}, "
            f"current={data.get('Present_Current', 'n/a')}, "
            f"load={data.get('Present_Load', 'n/a')}"
        )

        voltage_samples = samples.get(motor, [])
        if voltage_samples:
            nonzero_statuses = sorted(
                {status for status in status_samples.get(motor, []) if status != 0}
            )
            print(
                f"    sampled_voltage={min(voltage_samples) / 10:.1f}-"
                f"{max(voltage_samples) / 10:.1f} V "
                f"(n={len(voltage_samples)}), "
                f"nonzero_statuses={nonzero_statuses or 'none'}"
            )

    return 0


def motor_health_command(args: argparse.Namespace) -> int:
    return motor_health(args.port, retries=args.retries, seconds=args.seconds, hz=args.hz)


def dataset_root_for_repo(repo_id: str) -> Path:
    _, _, name = repo_id.rpartition("/")
    if name.startswith("eval_"):
        return DEFAULT_EVAL_DATASET_DIR
    return DEFAULT_TRAIN_DATASET_DIR


def lerobot_dataset_path(repo_id: str, dataset_root: str | Path | None = None) -> Path:
    root = Path(dataset_root) if dataset_root else dataset_root_for_repo(repo_id)
    return root / Path(repo_id)


def find_lerobot_dataset_path(repo_id: str) -> Path:
    preferred_path = lerobot_dataset_path(repo_id)
    if preferred_path.exists():
        return preferred_path

    candidates = [
        DEFAULT_TRAIN_DATASET_DIR / Path(repo_id),
        DEFAULT_EVAL_DATASET_DIR / Path(repo_id),
        DEFAULT_DATASET_DIR / Path(repo_id),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    return preferred_path


def remove_existing_dataset(repo_id: str) -> None:
    dataset_path = lerobot_dataset_path(repo_id).resolve()
    dataset_root = DEFAULT_DATASET_DIR.resolve()

    if dataset_root not in dataset_path.parents:
        raise RuntimeError(f"Refusing to remove path outside project datasets: {dataset_path}")

    if dataset_path.exists():
        print(f"Removing existing local dataset: {dataset_path}")
        shutil.rmtree(dataset_path)


def _record_fps(camera_config_path: Path, fps: int | None) -> int:
    if fps is not None:
        return fps

    if not camera_config_path.exists():
        return 30

    camera_fps_values = [
        int(camera["fps"])
        for camera in cameras.load_camera_config(camera_config_path)
        if camera.get("fps") is not None
    ]
    if not camera_fps_values:
        return 30

    return min(camera_fps_values)


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
    fps: int | None = None,
    display_data: bool = True,
    push_to_hub: bool = False,
    resume: bool = False,
    overwrite: bool = False,
    encoder_threads: int = 2,
    vcodec: str = "h264_nvenc",
    wait_start: bool = True,
) -> int:
    leader_id = leader_id or "leader"
    follower_id = follower_id or "follower"
    calib_dir = calibration_dir(calibration_dir_value)
    camera_config_path = Path(camera_config or cameras.DEFAULT_CONFIG_PATH)
    dataset_root_path = lerobot_dataset_path(repo_id, dataset_root)
    dataset_fps = _record_fps(camera_config_path, fps)
    camera_name_map = _base_smolvla_camera_name_map(camera_config_path)

    if overwrite and resume:
        raise RuntimeError("--overwrite and --resume cannot be used together")

    if overwrite:
        remove_existing_dataset(repo_id)

    print("Record controls:")
    print("  Right arrow: finish the current episode/reset early")
    print("  Left arrow : discard and rerecord the current episode")
    print("  Esc        : stop recording")
    if camera_name_map:
        print(f"Using base SmolVLA camera names: {json.dumps(camera_name_map)}")

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
            f"--robot.cameras={cameras.lerobot_cameras_arg(camera_config_path, camera_name_map)}",
            f"--display_data={str(display_data).lower()}",
            "--play_sounds=false",
            f"--dataset.repo_id={repo_id}",
            f"--dataset.root={dataset_root_path}",
            f"--dataset.num_episodes={episodes}",
            f"--dataset.single_task={task}",
            f"--dataset.fps={dataset_fps}",
            f"--dataset.episode_time_s={episode_time_s}",
            f"--dataset.reset_time_s={reset_time_s}",
            f"--dataset.push_to_hub={str(push_to_hub).lower()}",
            f"--resume={str(resume).lower()}",
            "--dataset.streaming_encoding=true",
            f"--dataset.encoder_threads={encoder_threads}",
            f"--dataset.vcodec={vcodec}",
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
        fps=args.fps,
        display_data=args.display_data,
        push_to_hub=args.push_to_hub,
        resume=args.resume,
        overwrite=args.overwrite,
        encoder_threads=args.encoder_threads,
        vcodec=args.vcodec,
        wait_start=args.wait_start,
    )


def rollout_policy(
    follower_port: str,
    policy_path: str = DEFAULT_SMOLVLA_POLICY_PATH,
    repo_id: str = DEFAULT_SMOLVLA_ROLLOUT_REPO_ID,
    task: str = "Pick up the object",
    leader_port: str | None = None,
    leader_id: str | None = None,
    follower_id: str | None = None,
    calibration_dir_value: str | None = None,
    camera_config: str | None = None,
    dataset_root: str | None = None,
    episodes: int = 3,
    episode_time_s: int = 30,
    reset_time_s: int = 15,
    fps: int = 30,
    display_data: bool = True,
    overwrite: bool = False,
    encoder_threads: int = 2,
    vcodec: str = "auto",
    device: str = "cuda",
    use_amp: bool = True,
    policy_num_steps: int | None = None,
    policy_num_vlm_layers: int | None = None,
    max_relative_target: float | None = 5.0,
    disable_torque_on_disconnect: bool = False,
    park_on_exit: bool = True,
    park_on_reset: bool = True,
    park_duration_s: float = 3.0,
    interpolation_multiplier: int = 1,
    wait_start: bool = True,
) -> int:
    leader_id = leader_id or "leader"
    follower_id = follower_id or "follower"
    repo_id = _policy_eval_repo_id(repo_id)
    calib_dir = calibration_dir(calibration_dir_value)
    camera_config_path = Path(camera_config or cameras.DEFAULT_CONFIG_PATH)
    dataset_root_path = lerobot_dataset_path(repo_id, dataset_root)
    camera_name_map = _base_smolvla_camera_name_map(camera_config_path)

    if overwrite:
        remove_existing_dataset(repo_id)

    print("Rollout controls:")
    print("  Right arrow: finish the current episode/reset early")
    print("  Left arrow : discard and rerecord the current episode")
    print("  Esc        : stop rollout")
    if camera_name_map:
        print(f"Using base SmolVLA camera names: {json.dumps(camera_name_map)}")

    if wait_start:
        os.environ["SO101_CAMERA_CONFIG"] = str(camera_config_path)
        os.environ["SO101_PARK_ON_EXIT"] = str(park_on_exit).lower()
        os.environ["SO101_PARK_ON_RESET"] = str(park_on_reset).lower()
        os.environ["SO101_PARK_DURATION_S"] = str(park_duration_s)
        command = [sys.executable, "-m", "svla.record_wait_start"]
    else:
        command = ["lerobot-record"]

    command.extend(
        [
            "--robot.type=so101_follower",
            f"--robot.port={follower_port}",
            f"--robot.id={follower_id}",
            f"--robot.calibration_dir={calib_dir}",
            f"--robot.cameras={cameras.lerobot_cameras_arg(camera_config_path, camera_name_map)}",
            f"--robot.disable_torque_on_disconnect={str(disable_torque_on_disconnect).lower()}",
            f"--policy.path={policy_path}",
            f"--policy.device={device}",
            f"--policy.use_amp={str(use_amp).lower()}",
            f"--display_data={str(display_data).lower()}",
            "--play_sounds=false",
            f"--dataset.repo_id={repo_id}",
            f"--dataset.root={dataset_root_path}",
            f"--dataset.num_episodes={episodes}",
            f"--dataset.single_task={task}",
            f"--dataset.fps={fps}",
            f"--dataset.episode_time_s={episode_time_s}",
            f"--dataset.reset_time_s={reset_time_s}",
            "--dataset.push_to_hub=false",
            "--dataset.streaming_encoding=true",
            f"--dataset.encoder_threads={encoder_threads}",
            f"--dataset.vcodec={vcodec}",
            f"--interpolation_multiplier={interpolation_multiplier}",
        ]
    )

    if policy_num_steps is not None:
        command.append(f"--policy.num_steps={policy_num_steps}")
    if policy_num_vlm_layers is not None:
        command.append(f"--policy.num_vlm_layers={policy_num_vlm_layers}")

    if max_relative_target is not None:
        command.append(f"--robot.max_relative_target={max_relative_target}")

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


def smolvla_check_command(_: argparse.Namespace) -> int:
    required_modules = {
        "accelerate": "accelerate",
        "num2words": "num2words",
        "safetensors": "safetensors",
        "transformers": "transformers",
        "lerobot.policies.smolvla": "lerobot[smolvla]",
    }
    missing = []
    for module, package in required_modules.items():
        if importlib.util.find_spec(module) is None:
            missing.append(package)

    if missing:
        print("Missing SmolVLA dependencies:")
        for package in missing:
            print(f"  - {package}")
        print("Run: uv sync")
        return 1

    print("SmolVLA dependencies are available.")
    print(f"Default policy path: {DEFAULT_SMOLVLA_POLICY_PATH}")
    return 0


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
        fps=args.fps,
        display_data=args.display_data,
        overwrite=args.overwrite,
        encoder_threads=args.encoder_threads,
        vcodec=args.vcodec,
        device=args.device,
        use_amp=args.use_amp,
        policy_num_steps=args.policy_num_steps,
        policy_num_vlm_layers=args.policy_num_vlm_layers,
        max_relative_target=args.max_relative_target,
        disable_torque_on_disconnect=args.disable_torque_on_disconnect,
        park_on_exit=args.park_on_exit,
        park_on_reset=args.park_on_reset,
        park_duration_s=args.park_duration_s,
        interpolation_multiplier=args.interpolation_multiplier,
        wait_start=args.wait_start,
    )


def dataset_path_command(args: argparse.Namespace) -> int:
    print(find_lerobot_dataset_path(args.repo_id))
    return 0


def dataset_open_command(args: argparse.Namespace) -> int:
    dataset_path = find_lerobot_dataset_path(args.repo_id)
    print(dataset_path)

    if not dataset_path.exists():
        print("Dataset path does not exist.")
        return 1

    if os.name == "nt":
        subprocess.Popen(["explorer", str(dataset_path)])
        return 0

    return run(["xdg-open", str(dataset_path)])


def dataset_info_command(args: argparse.Namespace) -> int:
    dataset_path = find_lerobot_dataset_path(args.repo_id)
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.exists():
        print(f"Dataset info not found: {info_path}")
        return 1

    info = json.loads(info_path.read_text(encoding="utf-8"))
    video_keys = [
        key for key, value in info.get("features", {}).items() if value.get("dtype") == "video"
    ]

    print(f"repo_id: {args.repo_id}")
    print(f"path: {dataset_path}")
    print(f"total_episodes: {info.get('total_episodes')}")
    print(f"total_frames: {info.get('total_frames')}")
    print(f"fps: {info.get('fps')}")
    print(f"splits: {info.get('splits')}")
    print(f"video_keys: {', '.join(video_keys)}")
    return 0


def _safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_").replace(":", "_")


def _base_smolvla_camera_name_map(camera_config_path: Path) -> dict[str, str]:
    if not camera_config_path.exists():
        return {}

    camera_configs = cameras.load_camera_config(camera_config_path)
    base_names = {"camera1", "camera2", "camera3"}
    camera_names = [str(camera["name"]) for camera in camera_configs]
    if all(name in base_names for name in camera_names):
        return {}

    used_targets = {name for name in camera_names if name in base_names}
    name_map: dict[str, str] = {}
    preferred_targets = {
        "side": ("camera1", "camera3"),
        "front": ("camera1", "camera3"),
        "upper": ("camera1", "camera3"),
        "top": ("camera1", "camera3"),
        "wrist": ("camera2",),
    }

    for camera in camera_configs:
        source_name = str(camera["name"])
        explicit_name = camera.get("feature_name")
        if explicit_name is None:
            continue

        target_name = str(explicit_name)
        if target_name != source_name:
            name_map[source_name] = target_name
        used_targets.add(target_name)

    for camera in camera_configs:
        source_name = str(camera["name"])
        if source_name in name_map or source_name in base_names:
            continue

        target_name = next(
            (
                candidate
                for candidate in preferred_targets.get(source_name, ())
                if candidate not in used_targets
            ),
            None,
        )
        if target_name is None:
            target_name = next(
                (
                    f"camera{camera_index}"
                    for camera_index in range(1, 4)
                    if f"camera{camera_index}" not in used_targets
                ),
                None,
            )
        if target_name is None:
            continue

        name_map[source_name] = target_name
        used_targets.add(target_name)

    return name_map


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


def _resolve_policy_pretrained_path(pretrained_path: str | None) -> str | None:
    if not pretrained_path:
        return None

    path = Path(pretrained_path)
    if path.exists():
        return str(path)

    if "/" not in pretrained_path:
        return pretrained_path

    from huggingface_hub import snapshot_download

    try:
        return snapshot_download(repo_id=pretrained_path, local_files_only=True)
    except Exception:
        return snapshot_download(repo_id=pretrained_path)


def _resolve_resume_config_path(output_dir_path: Path, config_path: str | None) -> str:
    if config_path:
        return str(Path(config_path))

    checkpoints_dir = output_dir_path / "checkpoints"
    if not checkpoints_dir.exists():
        raise FileNotFoundError(f"No checkpoints directory found: {checkpoints_dir}")

    checkpoint_dirs = sorted(
        path for path in checkpoints_dir.iterdir() if path.is_dir() and path.name.isdigit()
    )
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No numeric checkpoints found in: {checkpoints_dir}")

    train_config_path = checkpoint_dirs[-1] / "pretrained_model" / "train_config.json"
    if not train_config_path.exists():
        raise FileNotFoundError(f"Checkpoint train config not found: {train_config_path}")

    return str(train_config_path)


def train_policy(
    repo_id: str,
    dataset_root: str | None = None,
    policy: str = "smolvla",
    pretrained_path: str | None = DEFAULT_SMOLVLA_POLICY_PATH,
    device: str = "cuda",
    output_dir: str | None = None,
    job_name: str | None = None,
    resume: bool = False,
    config_path: str | None = None,
    steps: int = 3000,
    batch_size: int = 8,
    log_freq: int = 200,
    save_freq: int = 1000,
    eval_freq: int = 0,
    use_amp: bool = True,
    wandb: bool = False,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_mode: str | None = None,
    push_to_hub: bool = False,
    policy_repo_id: str | None = None,
    image_aug: bool = False,
    image_aug_max_num_transforms: int = 3,
    image_aug_random_order: bool = False,
) -> int:
    dataset_root_path = lerobot_dataset_path(repo_id, dataset_root)
    run_name = job_name or f"{policy}_{_safe_name(repo_id)}"
    output_dir_path = Path(output_dir) if output_dir else DEFAULT_TRAIN_OUTPUT_DIR / run_name
    resolved_pretrained_path = _resolve_policy_pretrained_path(pretrained_path)
    resolved_config_path = (
        _resolve_resume_config_path(output_dir_path, config_path) if resume else None
    )

    command = [
        sys.executable,
        "-m",
        "svla.train_compat",
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
        f"--log_freq={log_freq}",
        f"--save_freq={save_freq}",
        f"--eval_freq={eval_freq}",
        f"--wandb.enable={str(wandb).lower()}",
        "--wandb.disable_artifact=true",
    ]
    if resume:
        command.append("--resume=true")
    if resolved_config_path:
        command.append(f"--config_path={resolved_config_path}")

    if resolved_pretrained_path:
        command.append(f"--policy.pretrained_path={resolved_pretrained_path}")
    if image_aug:
        command.extend(
            [
                "--dataset.image_transforms.enable=true",
                f"--dataset.image_transforms.max_num_transforms={image_aug_max_num_transforms}",
                f"--dataset.image_transforms.random_order={str(image_aug_random_order).lower()}",
            ]
        )
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
        pretrained_path=args.pretrained_path,
        device=args.device,
        output_dir=args.output_dir,
        job_name=args.job_name,
        resume=args.resume,
        config_path=args.config_path,
        steps=args.steps,
        batch_size=args.batch_size,
        log_freq=args.log_freq,
        save_freq=args.save_freq,
        eval_freq=args.eval_freq,
        use_amp=args.use_amp,
        wandb=args.wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_mode=args.wandb_mode,
        push_to_hub=args.push_to_hub,
        policy_repo_id=args.policy_repo_id,
        image_aug=args.image_aug,
        image_aug_max_num_transforms=args.image_aug_max_num_transforms,
        image_aug_random_order=args.image_aug_random_order,
    )


def replay_episode(
    follower_port: str,
    repo_id: str,
    episode: int,
    dataset_root: str | None = None,
    follower_id: str | None = None,
    calibration_dir_value: str | None = None,
) -> int:
    follower_id = follower_id or "follower"
    calib_dir = calibration_dir(calibration_dir_value)
    dataset_root_path = lerobot_dataset_path(repo_id, dataset_root)

    return run(
        [
            "lerobot-replay",
            "--robot.type=so101_follower",
            f"--robot.port={follower_port}",
            f"--robot.id={follower_id}",
            f"--robot.calibration_dir={calib_dir}",
            f"--dataset.repo_id={repo_id}",
            f"--dataset.root={dataset_root_path}",
            f"--dataset.episode={episode}",
        ]
    )


def replay_command(args: argparse.Namespace) -> int:
    return replay_episode(
        follower_port=args.follower_port,
        repo_id=args.repo_id,
        episode=args.episode,
        dataset_root=args.dataset_root,
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

    relax_parser = subparsers.add_parser("relax", aliases=["unlock"])
    relax_parser.add_argument("--follower-port", required=True)
    relax_parser.add_argument("--follower-id")
    relax_parser.add_argument("--calibration-dir")
    relax_parser.set_defaults(func=relax_command)

    motor_scan_parser = subparsers.add_parser("motor-scan")
    motor_scan_parser.add_argument("--port", required=True)
    motor_scan_parser.add_argument("--retries", type=int, default=5)
    motor_scan_parser.set_defaults(func=motor_scan_command)

    motor_read_test_parser = subparsers.add_parser("motor-read-test")
    motor_read_test_parser.add_argument("--port", required=True)
    motor_read_test_parser.add_argument("--seconds", type=float, default=10.0)
    motor_read_test_parser.add_argument("--hz", type=float, default=20.0)
    motor_read_test_parser.add_argument("--retries", type=int, default=5)
    motor_read_test_parser.set_defaults(func=motor_read_test_command)

    motor_health_parser = subparsers.add_parser("motor-health")
    motor_health_parser.add_argument("--port", required=True)
    motor_health_parser.add_argument("--retries", type=int, default=5)
    motor_health_parser.add_argument("--seconds", type=float, default=0.0)
    motor_health_parser.add_argument("--hz", type=float, default=10.0)
    motor_health_parser.set_defaults(func=motor_health_command)

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
    record_parser.add_argument("--fps", type=int)
    record_parser.add_argument("--encoder-threads", type=int, default=2)
    record_parser.add_argument("--vcodec", default="h264_nvenc")
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
    rollout_parser.add_argument("--policy-path", default=DEFAULT_SMOLVLA_POLICY_PATH)
    rollout_parser.add_argument("--repo-id", default=DEFAULT_SMOLVLA_ROLLOUT_REPO_ID)
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
    rollout_parser.add_argument("--fps", type=int, default=30)
    rollout_parser.add_argument("--encoder-threads", type=int, default=2)
    rollout_parser.add_argument("--vcodec", default="auto")
    rollout_parser.add_argument("--device", default="cuda")
    rollout_parser.add_argument("--no-use-amp", action="store_false", dest="use_amp")
    rollout_parser.add_argument("--policy-num-steps", type=int)
    rollout_parser.add_argument("--policy-num-vlm-layers", type=int)
    rollout_parser.add_argument("--max-relative-target", type=float, default=5.0)
    rollout_parser.add_argument(
        "--no-max-relative-target",
        action="store_const",
        const=None,
        dest="max_relative_target",
    )
    rollout_parser.add_argument("--disable-torque-on-disconnect", action="store_true")
    rollout_parser.add_argument("--no-park-on-exit", action="store_false", dest="park_on_exit")
    rollout_parser.add_argument("--no-park-on-reset", action="store_false", dest="park_on_reset")
    rollout_parser.add_argument("--park-duration-s", type=float, default=3.0)
    rollout_parser.add_argument("--interpolation-multiplier", type=int, default=1)
    rollout_parser.add_argument("--overwrite", action="store_true")
    rollout_parser.add_argument("--no-wait-start", action="store_false", dest="wait_start")
    display_group = rollout_parser.add_mutually_exclusive_group()
    display_group.add_argument("--display-data", action="store_true", dest="display_data")
    display_group.add_argument("--no-display-data", action="store_false", dest="display_data")
    rollout_parser.set_defaults(display_data=True)
    rollout_parser.set_defaults(use_amp=True)
    rollout_parser.set_defaults(wait_start=True)
    rollout_parser.set_defaults(park_on_exit=True)
    rollout_parser.set_defaults(park_on_reset=True)
    rollout_parser.set_defaults(func=rollout_command)

    smolvla_check_parser = subparsers.add_parser("smolvla-check")
    smolvla_check_parser.set_defaults(func=smolvla_check_command)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--repo-id", required=True)
    train_parser.add_argument("--dataset-root")
    train_parser.add_argument("--policy", default="smolvla")
    train_parser.add_argument("--pretrained-path", default=DEFAULT_SMOLVLA_POLICY_PATH)
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--output-dir")
    train_parser.add_argument("--job-name")
    train_parser.add_argument("--resume", action="store_true")
    train_parser.add_argument("--config-path")
    train_parser.add_argument("--steps", type=int, default=3000)
    train_parser.add_argument("--batch-size", type=int, default=8)
    train_parser.add_argument("--log-freq", type=int, default=200)
    train_parser.add_argument("--save-freq", type=int, default=1000)
    train_parser.add_argument("--eval-freq", type=int, default=0)
    train_parser.add_argument("--no-use-amp", action="store_false", dest="use_amp")
    train_parser.add_argument("--wandb", action="store_true")
    train_parser.add_argument("--wandb-project")
    train_parser.add_argument("--wandb-entity")
    train_parser.add_argument("--wandb-mode")
    train_parser.add_argument("--push-to-hub", action="store_true")
    train_parser.add_argument("--policy-repo-id")
    train_parser.add_argument("--image-aug", action="store_true")
    train_parser.add_argument("--image-aug-max-num-transforms", type=int, default=3)
    train_parser.add_argument("--image-aug-random-order", action="store_true")
    train_parser.set_defaults(use_amp=True)
    train_parser.set_defaults(func=train_command)

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
    replay_parser.add_argument("--dataset-root")
    replay_parser.add_argument("--episode", type=int, default=0)
    replay_parser.add_argument("--follower-id")
    replay_parser.add_argument("--calibration-dir")
    replay_parser.set_defaults(func=replay_command)
