from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Any

import cv2
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.scripts import lerobot_record as lr

DEFAULT_MOTOR_NUM_RETRY = 3
DEFAULT_PARK_POSE = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -102.5,
    "elbow_flex": 94.0,
    "wrist_flex": 71.0,
    "wrist_roll": -1.0,
    "gripper": 1.2,
}


def _load_camera_settings() -> dict[str, dict[str, Any]]:
    config_path = os.environ.get("SO101_CAMERA_CONFIG")
    if not config_path:
        return {}

    path = Path(config_path)
    if not path.exists():
        return {}

    config = json.loads(path.read_text(encoding="utf-8"))
    cameras = config.get("cameras", [])
    return {str(camera.get("index_or_path", camera.get("index"))): camera for camera in cameras}


def install_camera_settings() -> None:
    camera_settings = _load_camera_settings()
    if not camera_settings:
        return

    original_connect = OpenCVCamera.connect

    def connect_with_settings(self: OpenCVCamera, *args: Any, **kwargs: Any) -> Any:
        result = original_connect(self, *args, **kwargs)
        setting = camera_settings.get(str(self.config.index_or_path))

        if setting is not None and self.videocapture is not None:
            name = setting.get("name", self.config.index_or_path)

            if "zoom" in setting:
                success = self.videocapture.set(cv2.CAP_PROP_ZOOM, float(setting["zoom"]))
                actual = self.videocapture.get(cv2.CAP_PROP_ZOOM)
                print(
                    f"{name}: record set zoom={setting['zoom']} "
                    f"(success={success}, actual={actual})"
                )

            focus_lock_after_s = setting.get("focus_lock_after_s")
            if focus_lock_after_s is not None:
                seconds = float(focus_lock_after_s)
                success = self.videocapture.set(cv2.CAP_PROP_AUTOFOCUS, 1)
                actual = self.videocapture.get(cv2.CAP_PROP_AUTOFOCUS)
                print(
                    f"{name}: record enable autofocus before focus lock "
                    f"(success={success}, actual_autofocus={actual})"
                )
                print(f"{name}: record waiting {seconds:.1f}s before locking autofocus")
                time.sleep(seconds)
                focus_value = self.videocapture.get(cv2.CAP_PROP_FOCUS)
                autofocus_success = self.videocapture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                focus_success = self.videocapture.set(cv2.CAP_PROP_FOCUS, focus_value)
                locked_focus = self.videocapture.get(cv2.CAP_PROP_FOCUS)
                actual_autofocus = self.videocapture.get(cv2.CAP_PROP_AUTOFOCUS)
                print(
                    f"{name}: record focus lock value={focus_value} "
                    f"(autofocus_success={autofocus_success}, "
                    f"actual_autofocus={actual_autofocus}, "
                    f"focus_success={focus_success}, actual_focus={locked_focus})"
                )

        return result

    OpenCVCamera.connect = connect_with_settings


def install_motor_bus_retries() -> None:
    num_retry = int(os.environ.get("SO101_MOTOR_NUM_RETRY", DEFAULT_MOTOR_NUM_RETRY))
    if num_retry <= 0:
        return

    from lerobot.motors.feetech import FeetechMotorsBus

    original_write = FeetechMotorsBus.write
    original_sync_read = FeetechMotorsBus.sync_read
    original_sync_write = FeetechMotorsBus.sync_write

    def write_with_retries(
        self: FeetechMotorsBus,
        data_name: str,
        motor: str,
        value: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        kwargs.setdefault("num_retry", num_retry)
        return original_write(self, data_name, motor, value, *args, **kwargs)

    def sync_read_with_retries(
        self: FeetechMotorsBus,
        data_name: str,
        motors: Any = None,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        kwargs.setdefault("num_retry", num_retry)
        return original_sync_read(self, data_name, motors, *args, **kwargs)

    def sync_write_with_retries(
        self: FeetechMotorsBus,
        data_name: str,
        values: Any,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        kwargs.setdefault("num_retry", num_retry)
        return original_sync_write(self, data_name, values, *args, **kwargs)

    FeetechMotorsBus.write = write_with_retries
    FeetechMotorsBus.sync_read = sync_read_with_retries
    FeetechMotorsBus.sync_write = sync_write_with_retries


def install_park_on_disconnect() -> None:
    if os.environ.get("SO101_PARK_ON_EXIT", "false").lower() != "true":
        return

    from lerobot.robots.so_follower.so_follower import SOFollower

    original_disconnect = SOFollower.disconnect

    def disconnect_with_park(self: SOFollower, *args: Any, **kwargs: Any) -> Any:
        if self.bus.is_connected:
            _park_follower(self)
        return original_disconnect(self, *args, **kwargs)

    SOFollower.disconnect = disconnect_with_park


def _park_follower(robot: Any) -> None:
    pose = _load_park_pose()
    duration_s = float(os.environ.get("SO101_PARK_DURATION_S", "3.0"))
    steps = max(1, int(duration_s * 20))

    try:
        present = robot.bus.sync_read("Present_Position")
    except Exception:
        logging.exception("Failed to read follower pose before parking.")
        return

    motors = [motor for motor in robot.bus.motors if motor in pose and motor in present]
    if not motors:
        return

    logging.info("Parking follower before disconnect.")
    for step in range(1, steps + 1):
        ratio = step / steps
        target = {
            motor: present[motor] + (float(pose[motor]) - present[motor]) * ratio
            for motor in motors
        }
        try:
            robot.bus.sync_write("Goal_Position", target)
        except Exception:
            logging.exception("Failed to send follower park pose.")
            return
        time.sleep(duration_s / steps)


def _load_park_pose() -> dict[str, float]:
    pose_text = os.environ.get("SO101_PARK_POSE")
    if not pose_text:
        return DEFAULT_PARK_POSE

    try:
        pose = json.loads(pose_text)
    except json.JSONDecodeError:
        logging.exception("Invalid SO101_PARK_POSE JSON; using default park pose.")
        return DEFAULT_PARK_POSE

    return {str(key): float(value) for key, value in pose.items()}


def install_wait_before_first_episode() -> None:
    original_record_loop = lr.record_loop
    original_log_say = lr.log_say
    waiting_done = False
    pending_recording_message: tuple[str, bool] | None = None

    def log_say_with_initial_wait(
        message: str, play_sounds: bool, *args: Any, **kwargs: Any
    ) -> Any:
        nonlocal pending_recording_message

        if not waiting_done and message.startswith("Recording episode"):
            pending_recording_message = (message, play_sounds)
            return None

        return original_log_say(message, play_sounds, *args, **kwargs)

    def record_loop_with_initial_wait(*args: Any, **kwargs: Any) -> Any:
        nonlocal pending_recording_message, waiting_done

        dataset = kwargs.get("dataset")
        events = kwargs.get("events")

        if not waiting_done and dataset is not None and events is not None:
            waiting_done = True
            lr.log_say("Reset the environment. Press right arrow to start episode 0.", False)
            _run_initial_wait_loop(original_record_loop, kwargs)
            events["exit_early"] = False
            if events["stop_recording"]:
                return None

            if pending_recording_message is not None:
                message, play_sounds = pending_recording_message
                original_log_say(message, play_sounds)
                pending_recording_message = None

        return original_record_loop(*args, **kwargs)

    lr.log_say = log_say_with_initial_wait
    lr.record_loop = record_loop_with_initial_wait


def _run_initial_wait_loop(original_record_loop: Any, kwargs: dict[str, Any]) -> Any:
    original_warning = logging.warning

    def warning_without_expected_no_action(message: object, *args: Any, **kwargs: Any) -> Any:
        if isinstance(message, str) and message.startswith("No policy or teleoperator provided"):
            return None
        return original_warning(message, *args, **kwargs)

    logging.warning = warning_without_expected_no_action
    try:
        return original_record_loop(
            **{
                **kwargs,
                "dataset": None,
                "policy": None,
                "preprocessor": None,
                "postprocessor": None,
                "interpolator": None,
                "control_time_s": 3600,
            }
        )
    finally:
        logging.warning = original_warning


def main() -> None:
    install_motor_bus_retries()
    install_camera_settings()
    install_park_on_disconnect()
    install_wait_before_first_episode()
    lr.main()


if __name__ == "__main__":
    main()
