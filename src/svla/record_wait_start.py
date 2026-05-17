from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import cv2
from lerobot.cameras.opencv.camera_opencv import OpenCVCamera
from lerobot.scripts import lerobot_record as lr


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
            lr.log_say("Reset the environment. Press right arrow to start episode 0.", True)
            original_record_loop(
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


def main() -> None:
    install_camera_settings()
    install_wait_before_first_episode()
    lr.main()


if __name__ == "__main__":
    main()
