from __future__ import annotations

import argparse
import html
import http.server
import json
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "cameras" / "camera_config.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "camera_checks"


def load_camera_config(config_path: Path) -> list[dict[str, Any]]:
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    cameras = config.get("cameras")
    if not isinstance(cameras, list) or not cameras:
        raise ValueError(f"{config_path} must contain a non-empty 'cameras' list")

    return cameras


def lerobot_cameras_arg(config_path: Path) -> str:
    cameras = load_camera_config(config_path)
    camera_items = []

    for camera in cameras:
        name = camera["name"]
        index_or_path = camera.get("index_or_path", camera.get("index"))
        width = camera["width"]
        height = camera["height"]
        fps = camera["fps"]
        camera_type = camera.get("type", "opencv")

        camera_items.append(
            f"{name}: {{type: {camera_type}, index_or_path: {index_or_path}, "
            f"width: {width}, height: {height}, fps: {fps}}}"
        )

    return "{ " + ", ".join(camera_items) + " }"


def _open_camera(
    index_or_path: int | str, width: int | None, height: int | None, fps: int | None
) -> cv2.VideoCapture:
    if isinstance(index_or_path, int):
        cap = cv2.VideoCapture(index_or_path, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index_or_path, cv2.CAP_ANY)

    if width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    if height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if fps is not None:
        cap.set(cv2.CAP_PROP_FPS, fps)
    return cap


def _set_camera_property(
    name: str,
    cap: cv2.VideoCapture,
    property_name: str,
    property_id: int,
    requested_value: Any,
) -> None:
    requested_float = float(requested_value)
    success = cap.set(property_id, requested_float)
    actual_value = cap.get(property_id)
    print(
        f"{name}: set {property_name}={requested_value} (success={success}, actual={actual_value})"
    )


def _read_for_seconds(cap: cv2.VideoCapture, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        cap.read()
        time.sleep(0.03)


def _lock_focus_after_autofocus(name: str, cap: cv2.VideoCapture, seconds: float) -> None:
    enable_success = cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
    actual_enabled = cap.get(cv2.CAP_PROP_AUTOFOCUS)
    print(
        f"{name}: enable autofocus before focus lock "
        f"(success={enable_success}, actual_autofocus={actual_enabled})"
    )

    print(f"{name}: waiting {seconds:.1f}s before locking autofocus")
    _read_for_seconds(cap, seconds)

    focus_value = cap.get(cv2.CAP_PROP_FOCUS)
    autofocus_success = cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    focus_success = cap.set(cv2.CAP_PROP_FOCUS, focus_value)
    locked_focus_value = cap.get(cv2.CAP_PROP_FOCUS)
    actual_autofocus = cap.get(cv2.CAP_PROP_AUTOFOCUS)

    print(
        f"{name}: focus lock value={focus_value} "
        f"(autofocus_success={autofocus_success}, actual_autofocus={actual_autofocus}, "
        f"focus_success={focus_success}, actual_focus={locked_focus_value})"
    )


def _camera_info(cap: cv2.VideoCapture) -> tuple[int, int, float]:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    return width, height, fps


def _camera_index_or_path(camera: dict[str, Any]) -> int | str:
    index_or_path = camera.get("index_or_path", camera.get("index"))
    if isinstance(index_or_path, str):
        return int(index_or_path) if index_or_path.isdigit() else index_or_path
    return int(index_or_path)


def _apply_camera_properties(name: str, cap: cv2.VideoCapture, camera: dict[str, Any]) -> None:
    camera_zoom = camera.get("zoom")
    focus_lock_after_s = camera.get("focus_lock_after_s")

    if camera_zoom is not None:
        _set_camera_property(name, cap, "zoom", cv2.CAP_PROP_ZOOM, camera_zoom)

    if focus_lock_after_s is not None:
        _lock_focus_after_autofocus(name, cap, float(focus_lock_after_s))


def _config_default(cameras: list[dict[str, Any]], key: str, fallback: Any) -> Any:
    for camera in cameras:
        value = camera.get(key)
        if value is not None:
            return value
    return fallback


def _scan_camera_specs(
    config_path: Path,
    scan_max: int,
    width: int | None,
    height: int | None,
    fps: int | None,
) -> list[dict[str, Any]]:
    configured_cameras = load_camera_config(config_path) if config_path.exists() else []
    configured_by_index = {}
    for camera in configured_cameras:
        index_or_path = _camera_index_or_path(camera)
        if isinstance(index_or_path, int):
            configured_by_index[index_or_path] = dict(camera)

    scan_width = _config_default(configured_cameras, "width", width)
    scan_height = _config_default(configured_cameras, "height", height)
    scan_fps = _config_default(configured_cameras, "fps", fps)
    camera_specs = []

    for index in range(scan_max + 1):
        camera = configured_by_index.get(index, {})
        camera_spec = {
            "name": f"index_{index}",
            "index": index,
            "width": scan_width,
            "height": scan_height,
            "fps": scan_fps,
        }
        camera_spec.update(camera)
        camera_spec["index"] = index
        camera_specs.append(camera_spec)

    return camera_specs


def _camera_specs(
    config_path: Path,
    scan: bool,
    scan_max: int,
    width: int | None,
    height: int | None,
    fps: int | None,
) -> list[dict[str, Any]]:
    if scan:
        return _scan_camera_specs(config_path, scan_max, width, height, fps)

    if config_path.exists():
        cameras = load_camera_config(config_path)
        for camera in cameras:
            camera.setdefault("width", width)
            camera.setdefault("height", height)
            camera.setdefault("fps", fps)
        return cameras

    return [
        {
            "name": "camera_0",
            "index": 0,
            "width": width,
            "height": height,
            "fps": fps,
        },
        {
            "name": "camera_1",
            "index": 1,
            "width": width,
            "height": height,
            "fps": fps,
        },
    ]


def check_cameras(
    cameras: list[dict[str, Any]],
    width: int | None,
    height: int | None,
    fps: int | None,
    output_dir: Path,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    captures: list[tuple[str, cv2.VideoCapture]] = []

    try:
        for camera in cameras:
            name = str(camera.get("name") or f"camera_{camera['index']}")
            index_or_path = _camera_index_or_path(camera)
            camera_width = camera.get("width", width)
            camera_height = camera.get("height", height)
            camera_fps = camera.get("fps", fps)

            cap = _open_camera(index_or_path, camera_width, camera_height, camera_fps)
            if not cap.isOpened():
                print(f"{name} ({index_or_path}): failed to open")
                continue

            _apply_camera_properties(name, cap, camera)

            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"{name} ({index_or_path}): opened, but failed to read a frame")
                cap.release()
                continue

            actual_width, actual_height, actual_fps = _camera_info(cap)
            image_path = output_dir / f"{name}_{timestamp}.jpg"
            cv2.imwrite(str(image_path), frame)
            print(
                f"{name} ({index_or_path}): {actual_width}x{actual_height} @ {actual_fps:.1f} fps, "
                f"snapshot={image_path}"
            )
            captures.append((name, cap))

        if not captures:
            print("No cameras were available.")
            return 1

        return 0
    finally:
        for _, cap in captures:
            cap.release()


def preview_cameras(
    cameras: list[dict[str, Any]],
    width: int | None,
    height: int | None,
    fps: int | None,
    output_dir: Path,
    host: str,
    port: int,
    preview_interval_ms: int,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    preview_interval_ms = max(50, preview_interval_ms)
    captures: list[dict[str, Any]] = []

    try:
        for camera in cameras:
            name = str(camera.get("name") or f"camera_{camera.get('index', 'unknown')}")
            index_or_path = _camera_index_or_path(camera)
            camera_width = camera.get("width", width)
            camera_height = camera.get("height", height)
            camera_fps = camera.get("fps", fps)

            cap = _open_camera(index_or_path, camera_width, camera_height, camera_fps)
            if not cap.isOpened():
                print(f"{name} ({index_or_path}): failed to open")
                continue

            _apply_camera_properties(name, cap, camera)
            actual_width, actual_height, actual_fps = _camera_info(cap)
            print(
                f"{name} ({index_or_path}): previewing "
                f"{actual_width}x{actual_height} @ {actual_fps:.1f} fps"
            )
            captures.append(
                {
                    "name": name,
                    "index_or_path": index_or_path,
                    "cap": cap,
                    "lock": threading.Lock(),
                    "width": actual_width,
                    "height": actual_height,
                    "fps": actual_fps,
                }
            )

        if not captures:
            print("No cameras were available.")
            return 1

        server = _preview_server(captures, output_dir, host, port, preview_interval_ms)
        print(f"Open preview: http://{host}:{server.server_port}")
        print(
            "Press Ctrl+C in this terminal to stop. "
            "Preview refreshes after each frame loads, "
            f"with {preview_interval_ms} ms between frames. "
            "Use the browser button to save snapshots."
        )
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            print("Stopping preview.")
        finally:
            server.server_close()
        return 0
    finally:
        for capture in captures:
            capture["cap"].release()


def _preview_server(
    captures: list[dict[str, Any]],
    output_dir: Path,
    host: str,
    port: int,
    preview_interval_ms: int,
) -> http.server.ThreadingHTTPServer:
    class PreviewHandler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def do_GET(self) -> None:
            try:
                parsed_path = urlparse(self.path)
                if parsed_path.path == "/":
                    return self._send_page()
                if parsed_path.path == "/snapshot":
                    return self._save_snapshots()
                if parsed_path.path.startswith("/camera/") and parsed_path.path.endswith(".jpg"):
                    camera_id_text = parsed_path.path.removeprefix("/camera/").removesuffix(".jpg")
                    return self._send_frame(camera_id_text)
                self.send_error(404)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return

        def _send_page(self) -> None:
            cards = []
            for camera_id, capture in enumerate(captures):
                name = html.escape(str(capture["name"]))
                index_or_path = str(capture["index_or_path"])
                width = int(capture["width"])
                height = int(capture["height"])
                fps = float(capture["fps"])
                camera_label = f"{index_or_path} - {width}x{height} @ {fps:.1f} fps"
                cards.append(
                    f"""
                    <section>
                      <h2>{name} <span>{html.escape(camera_label)}</span></h2>
                      <img src="/camera/{camera_id}.jpg" data-camera-id="{camera_id}" alt="{name}">
                    </section>
                    """
                )

            body = "\n".join(cards)
            page = f"""<!doctype html>
            <html lang="en">
            <head>
            <meta charset="utf-8">
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <title>SO101 Camera Preview</title>
            <style>
                body {{
                margin: 0;
                font-family: system-ui, sans-serif;
                background: #111;
                color: #f5f5f5;
                }}
                header {{
                position: sticky;
                top: 0;
                padding: 12px 16px;
                background: #181818;
                z-index: 1;
                }}
                button {{ padding: 8px 12px; font: inherit; }}
                main {{
                display: grid;
                gap: 16px;
                padding: 16px;
                grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
                }}
                h2 {{ font-size: 16px; margin: 0 0 8px; }}
                h2 span {{ color: #aaa; font-weight: 400; }}
                img {{ width: 100%; height: auto; background: #000; }}
            </style>
            </head>
            <body>
            <header><button onclick="saveSnapshots()">Save snapshots</button></header>
            <main>{body}</main>
            <script>
                const intervalMs = {preview_interval_ms};
                function refreshImage(img) {{
                    img.src = `/camera/${{img.dataset.cameraId}}.jpg?t=${{Date.now()}}`;
                }}
                function scheduleRefresh(img) {{
                    setTimeout(() => refreshImage(img), intervalMs);
                }}
                async function saveSnapshots() {{
                    await fetch('/snapshot');
                }}
                for (const img of document.querySelectorAll('img[data-camera-id]')) {{
                    img.addEventListener('load', () => scheduleRefresh(img));
                    img.addEventListener('error', () => scheduleRefresh(img));
                    if (img.complete) {{
                        scheduleRefresh(img);
                    }}
                }}
            </script>
            </body>
            </html>
            """
            page_bytes = page.encode("utf-8")
            self._send_bytes(page_bytes, "text/html; charset=utf-8")

        def _send_frame(self, camera_id_text: str) -> None:
            try:
                capture = captures[int(camera_id_text)]
            except (ValueError, IndexError):
                self.send_error(404)
                return

            frame = _read_frame(capture)
            if frame is None:
                self.send_error(503)
                return

            ok, encoded = cv2.imencode(".jpg", frame)
            if not ok:
                self.send_error(500)
                return

            frame_bytes = encoded.tobytes()
            self._send_bytes(frame_bytes, "image/jpeg", cache_control="no-store")

        def _save_snapshots(self) -> None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            saved_paths = []
            for capture in captures:
                frame = _read_frame(capture)
                if frame is None:
                    continue
                image_path = output_dir / f"{capture['name']}_{timestamp}.jpg"
                cv2.imwrite(str(image_path), frame)
                saved_paths.append(str(image_path))
                print(f"{capture['name']}: snapshot={image_path}")

            response = "\n".join(saved_paths).encode("utf-8")
            self._send_bytes(response, "text/plain; charset=utf-8")

        def _send_bytes(
            self,
            response_bytes: bytes,
            content_type: str,
            cache_control: str | None = None,
        ) -> None:
            try:
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                if cache_control is not None:
                    self.send_header("Cache-Control", cache_control)
                self.send_header("Content-Length", str(len(response_bytes)))
                self.end_headers()
                self.wfile.write(response_bytes)
            except (BrokenPipeError, ConnectionAbortedError, ConnectionResetError):
                return

    return http.server.ThreadingHTTPServer((host, port), PreviewHandler)


def _read_frame(capture: dict[str, Any]) -> Any | None:
    with capture["lock"]:
        ok, frame = capture["cap"].read()
    return frame if ok and frame is not None else None


def cameras_command(args: argparse.Namespace) -> int:
    cameras = _camera_specs(
        config_path=Path(args.config),
        scan=args.scan,
        scan_max=args.scan_max,
        width=args.width,
        height=args.height,
        fps=args.fps,
    )
    if args.preview:
        return preview_cameras(
            cameras=cameras,
            width=args.width,
            height=args.height,
            fps=args.fps,
            output_dir=Path(args.output_dir),
            host=args.preview_host,
            port=args.preview_port,
            preview_interval_ms=args.preview_interval_ms,
        )

    return check_cameras(
        cameras=cameras,
        width=args.width,
        height=args.height,
        fps=args.fps,
        output_dir=Path(args.output_dir),
    )


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("cameras")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--scan", action="store_true")
    parser.add_argument("--scan-max", type=int, default=5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--preview-host", default="127.0.0.1")
    parser.add_argument("--preview-port", type=int, default=8765)
    parser.add_argument("--preview-interval-ms", type=int, default=100)
    parser.set_defaults(func=cameras_command)
