import json
import math
import os
import subprocess
import threading
from tempfile import NamedTemporaryFile

from bsor.Bsor import (
    Bsor,
    ControllerOffsets,
    Cut,
    Frame,
    Height,
    Info,
    Note,
    Pause,
    UserData,
    VRObject,
    Wall,
    make_bsor,
)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
BSOR_TOOLS_DIR = os.path.join(REPO_ROOT, "rust", "bsor_tools")
BSOR_TOOLS_MANIFEST = os.path.join(BSOR_TOOLS_DIR, "Cargo.toml")
BSOR_TOOLS_RELEASE_BIN = os.path.join(
    BSOR_TOOLS_DIR,
    "target",
    "release",
    "bsor_tools.exe" if os.name == "nt" else "bsor_tools",
)
BSOR_TOOLS_DEBUG_BIN = os.path.join(
    BSOR_TOOLS_DIR,
    "target",
    "debug",
    "bsor_tools.exe" if os.name == "nt" else "bsor_tools",
)

_BUILD_LOCK = threading.Lock()
_BUILD_ATTEMPTED = False


def _normalize_backend(value, default="auto"):
    backend = str(value or default).strip().lower()
    if backend not in {"auto", "python", "rust"}:
        return default
    return backend


def _binary_candidates():
    configured = os.environ.get("CYBERNOODLES_BSOR_TOOLS_BIN", "").strip()
    candidates = []
    if configured:
        candidates.append(configured)
    candidates.append(BSOR_TOOLS_RELEASE_BIN)
    candidates.append(BSOR_TOOLS_DEBUG_BIN)
    return candidates


def _iter_bsor_tools_source_paths():
    for path in (BSOR_TOOLS_MANIFEST, os.path.join(BSOR_TOOLS_DIR, "Cargo.lock")):
        if os.path.isfile(path):
            yield path

    src_dir = os.path.join(BSOR_TOOLS_DIR, "src")
    if not os.path.isdir(src_dir):
        return
    for root, _, files in os.walk(src_dir):
        for filename in files:
            if filename.endswith(".rs"):
                yield os.path.join(root, filename)


def _latest_bsor_tools_source_mtime():
    latest = None
    for path in _iter_bsor_tools_source_paths():
        try:
            mtime = os.path.getmtime(path)
        except OSError:
            continue
        latest = mtime if latest is None else max(latest, mtime)
    return latest


def _binary_is_stale(binary):
    try:
        binary_mtime = os.path.getmtime(binary)
    except OSError:
        return True
    source_mtime = _latest_bsor_tools_source_mtime()
    return source_mtime is not None and source_mtime > (binary_mtime + 1e-6)


def _find_existing_binary(allow_stale=False):
    for candidate in _binary_candidates():
        if candidate and os.path.isfile(candidate) and (allow_stale or not _binary_is_stale(candidate)):
            return candidate
    return None


def _build_bsor_tools():
    global _BUILD_ATTEMPTED
    with _BUILD_LOCK:
        binary = _find_existing_binary()
        if binary:
            return binary
        if _BUILD_ATTEMPTED:
            return None
        _BUILD_ATTEMPTED = True
        if not os.path.isfile(BSOR_TOOLS_MANIFEST):
            return None
        try:
            print(f"Building Rust bsor_tools: {BSOR_TOOLS_MANIFEST}", flush=True)
            subprocess.run(
                ["cargo", "build", "--release", "--manifest-path", BSOR_TOOLS_MANIFEST],
                cwd=REPO_ROOT,
                check=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except Exception:
            return None
        return _find_existing_binary()


def resolve_bsor_tools_binary(auto_build=False):
    binary = _find_existing_binary()
    if binary:
        return binary
    if auto_build:
        return _build_bsor_tools()
    return None


def bsor_tools_available(auto_build=False):
    return bool(resolve_bsor_tools_binary(auto_build=auto_build))


def _run_bsor_tools(args, *, stdin_text=None, auto_build=False, stream_output=False):
    binary = resolve_bsor_tools_binary(auto_build=auto_build)
    if not binary:
        raise RuntimeError(
            "Rust bsor_tools binary is not available. Build it with "
            f"`cargo build --release --manifest-path \"{BSOR_TOOLS_MANIFEST}\"`."
        )

    if stream_output:
        completed = subprocess.run(
            [binary, *args],
            cwd=REPO_ROOT,
            input=stdin_text,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"bsor_tools {' '.join(args)} failed with exit code {completed.returncode}"
            )
        return ""

    completed = subprocess.run(
        [binary, *args],
        cwd=REPO_ROOT,
        input=stdin_text,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip() or "unknown error"
        raise RuntimeError(f"bsor_tools {' '.join(args)} failed: {detail}")
    return completed.stdout


def _vr_payload(obj):
    return {
        "x": float(getattr(obj, "x", 0.0)),
        "y": float(getattr(obj, "y", 0.0)),
        "z": float(getattr(obj, "z", 0.0)),
        "x_rot": float(getattr(obj, "x_rot", 0.0)),
        "y_rot": float(getattr(obj, "y_rot", 0.0)),
        "z_rot": float(getattr(obj, "z_rot", 0.0)),
        "w_rot": float(getattr(obj, "w_rot", 1.0)),
    }


def _cut_payload(cut):
    if cut is None:
        return None
    return {
        "speedOK": bool(getattr(cut, "speedOK", False)),
        "directionOk": bool(getattr(cut, "directionOk", False)),
        "saberTypeOk": bool(getattr(cut, "saberTypeOk", False)),
        "wasCutTooSoon": bool(getattr(cut, "wasCutTooSoon", False)),
        "saberSpeed": float(getattr(cut, "saberSpeed", 0.0)),
        "saberDirection": [
            float(value) for value in getattr(cut, "saberDirection", [0.0, 0.0, 0.0])
        ],
        "saberType": int(getattr(cut, "saberType", 0)),
        "timeDeviation": float(getattr(cut, "timeDeviation", 0.0)),
        "cutDeviation": float(getattr(cut, "cutDeviation", 0.0)),
        "cutPoint": [float(value) for value in getattr(cut, "cutPoint", [0.0, 0.0, 0.0])],
        "cutNormal": [float(value) for value in getattr(cut, "cutNormal", [0.0, 0.0, 0.0])],
        "cutDistanceToCenter": float(getattr(cut, "cutDistanceToCenter", 0.0)),
        "cutAngle": float(getattr(cut, "cutAngle", 0.0)),
        "beforeCutRating": float(getattr(cut, "beforeCutRating", 0.0)),
        "afterCutRating": float(getattr(cut, "afterCutRating", 0.0)),
    }


def _user_data_payload(item):
    key = getattr(item, "key", None)
    raw_bytes = getattr(item, "bytes", None)
    if key is None or raw_bytes is None:
        raise TypeError(
            "Unsupported BSOR user_data entry for Rust writer; expected key+bytes fields."
        )
    import base64

    return {
        "key": str(key),
        "bytes_base64": base64.b64encode(bytes(raw_bytes)).decode("ascii"),
    }


def bsor_to_payload(replay):
    info = getattr(replay, "info", None)
    if info is None:
        raise TypeError("BSOR object is missing info")

    controller_offsets = getattr(replay, "controller_offsets", None)
    if not getattr(controller_offsets, "left", None) or not getattr(controller_offsets, "right", None):
        controller_offsets = None

    return {
        "magic_number": int(getattr(replay, "magic_number", 0x442D3D69)),
        "file_version": int(getattr(replay, "file_version", 1)),
        "info": {
            "version": str(getattr(info, "version", "")),
            "gameVersion": str(getattr(info, "gameVersion", "")),
            "timestamp": str(getattr(info, "timestamp", "")),
            "playerId": str(getattr(info, "playerId", "")),
            "playerName": str(getattr(info, "playerName", "")),
            "platform": str(getattr(info, "platform", "")),
            "trackingSystem": str(getattr(info, "trackingSystem", "")),
            "hmd": str(getattr(info, "hmd", "")),
            "controller": str(getattr(info, "controller", "")),
            "songHash": str(getattr(info, "songHash", "")),
            "songName": str(getattr(info, "songName", "")),
            "mapper": str(getattr(info, "mapper", "")),
            "difficulty": str(getattr(info, "difficulty", "")),
            "score": int(getattr(info, "score", 0) or 0),
            "mode": str(getattr(info, "mode", "")),
            "environment": str(getattr(info, "environment", "")),
            "modifiers": str(getattr(info, "modifiers", "")),
            "jumpDistance": float(getattr(info, "jumpDistance", 0.0)),
            "leftHanded": bool(getattr(info, "leftHanded", False)),
            "height": float(getattr(info, "height", 0.0)),
            "startTime": float(getattr(info, "startTime", 0.0)),
            "failTime": float(getattr(info, "failTime", 0.0)),
            "speed": float(getattr(info, "speed", 0.0)),
        },
        "frames": [
            {
                "time": float(getattr(frame, "time", 0.0)),
                "fps": int(getattr(frame, "fps", 0) or 0),
                "head": _vr_payload(getattr(frame, "head", None)),
                "left_hand": _vr_payload(getattr(frame, "left_hand", None)),
                "right_hand": _vr_payload(getattr(frame, "right_hand", None)),
            }
            for frame in getattr(replay, "frames", [])
        ],
        "notes": [
            {
                "note_id": int(getattr(note, "note_id", 0) or 0),
                "scoringType": int(getattr(note, "scoringType", 0) or 0),
                "lineIndex": int(getattr(note, "lineIndex", 0) or 0),
                "noteLineLayer": int(getattr(note, "noteLineLayer", 0) or 0),
                "colorType": int(getattr(note, "colorType", 0) or 0),
                "cutDirection": int(getattr(note, "cutDirection", 0) or 0),
                "event_time": float(getattr(note, "event_time", 0.0)),
                "spawn_time": float(getattr(note, "spawn_time", 0.0)),
                "event_type": int(getattr(note, "event_type", 0) or 0),
                "cut": _cut_payload(getattr(note, "cut", None)),
                "pre_score": int(getattr(note, "pre_score", 0) or 0),
                "post_score": int(getattr(note, "post_score", 0) or 0),
                "acc_score": int(getattr(note, "acc_score", 0) or 0),
                "score": int(getattr(note, "score", 0) or 0),
            }
            for note in getattr(replay, "notes", [])
        ],
        "walls": [
            {
                "id": int(getattr(wall, "id", 0) or 0),
                "energy": float(getattr(wall, "energy", 0.0)),
                "time": float(getattr(wall, "time", 0.0)),
                "spawnTime": float(getattr(wall, "spawnTime", 0.0)),
            }
            for wall in getattr(replay, "walls", [])
        ],
        "heights": [
            {
                "height": float(getattr(height, "height", 0.0)),
                "time": float(getattr(height, "time", 0.0)),
            }
            for height in getattr(replay, "heights", [])
        ],
        "pauses": [
            {
                "duration": int(getattr(pause, "duration", 0) or 0),
                "time": float(getattr(pause, "time", 0.0)),
            }
            for pause in getattr(replay, "pauses", [])
        ],
        "controller_offsets": (
            None
            if controller_offsets is None
            else {
                "left": _vr_payload(getattr(controller_offsets, "left", None)),
                "right": _vr_payload(getattr(controller_offsets, "right", None)),
            }
        ),
        "user_data": [_user_data_payload(item) for item in getattr(replay, "user_data", [])],
    }


def _make_vr_object(payload):
    obj = VRObject()
    obj.x = float(payload.get("x", 0.0))
    obj.y = float(payload.get("y", 0.0))
    obj.z = float(payload.get("z", 0.0))
    obj.x_rot = float(payload.get("x_rot", 0.0))
    obj.y_rot = float(payload.get("y_rot", 0.0))
    obj.z_rot = float(payload.get("z_rot", 0.0))
    obj.w_rot = float(payload.get("w_rot", 1.0))
    return obj


def _make_cut(payload):
    if not payload:
        return None
    cut = Cut()
    cut.speedOK = bool(payload.get("speedOK", False))
    cut.directionOk = bool(payload.get("directionOk", False))
    cut.saberTypeOk = bool(payload.get("saberTypeOk", False))
    cut.wasCutTooSoon = bool(payload.get("wasCutTooSoon", False))
    cut.saberSpeed = float(payload.get("saberSpeed", 0.0))
    cut.saberDirection = [float(value) for value in payload.get("saberDirection", [0.0, 0.0, 0.0])]
    cut.saberType = int(payload.get("saberType", 0) or 0)
    cut.timeDeviation = float(payload.get("timeDeviation", 0.0))
    cut.cutDeviation = float(payload.get("cutDeviation", 0.0))
    cut.cutPoint = [float(value) for value in payload.get("cutPoint", [0.0, 0.0, 0.0])]
    cut.cutNormal = [float(value) for value in payload.get("cutNormal", [0.0, 0.0, 0.0])]
    cut.cutDistanceToCenter = float(payload.get("cutDistanceToCenter", 0.0))
    cut.cutAngle = float(payload.get("cutAngle", 0.0))
    cut.beforeCutRating = float(payload.get("beforeCutRating", 0.0))
    cut.afterCutRating = float(payload.get("afterCutRating", 0.0))
    return cut


def payload_to_bsor(payload):
    replay = Bsor()
    replay.magic_number = int(payload.get("magic_number", 0x442D3D69))
    replay.file_version = int(payload.get("file_version", 1))

    info_payload = payload.get("info", {}) or {}
    info = Info()
    info.version = str(info_payload.get("version", ""))
    info.gameVersion = str(info_payload.get("gameVersion", ""))
    info.timestamp = str(info_payload.get("timestamp", ""))
    info.playerId = str(info_payload.get("playerId", ""))
    info.playerName = str(info_payload.get("playerName", ""))
    info.platform = str(info_payload.get("platform", ""))
    info.trackingSystem = str(info_payload.get("trackingSystem", ""))
    info.hmd = str(info_payload.get("hmd", ""))
    info.controller = str(info_payload.get("controller", ""))
    info.songHash = str(info_payload.get("songHash", ""))
    info.songName = str(info_payload.get("songName", ""))
    info.mapper = str(info_payload.get("mapper", ""))
    info.difficulty = str(info_payload.get("difficulty", ""))
    info.score = int(info_payload.get("score", 0) or 0)
    info.mode = str(info_payload.get("mode", ""))
    info.environment = str(info_payload.get("environment", ""))
    info.modifiers = str(info_payload.get("modifiers", ""))
    info.jumpDistance = float(info_payload.get("jumpDistance", 0.0))
    info.leftHanded = bool(info_payload.get("leftHanded", False))
    info.height = float(info_payload.get("height", 0.0))
    info.startTime = float(info_payload.get("startTime", 0.0))
    info.failTime = float(info_payload.get("failTime", 0.0))
    info.speed = float(info_payload.get("speed", 0.0))
    replay.info = info

    replay.frames = []
    for frame_payload in payload.get("frames", []) or []:
        frame = Frame()
        frame.time = float(frame_payload.get("time", 0.0))
        frame.fps = int(frame_payload.get("fps", 0) or 0)
        frame.head = _make_vr_object(frame_payload.get("head", {}) or {})
        frame.left_hand = _make_vr_object(frame_payload.get("left_hand", {}) or {})
        frame.right_hand = _make_vr_object(frame_payload.get("right_hand", {}) or {})
        replay.frames.append(frame)

    replay.notes = []
    for note_payload in payload.get("notes", []) or []:
        note = Note()
        note.note_id = int(note_payload.get("note_id", 0) or 0)
        note.scoringType = int(note_payload.get("scoringType", 0) or 0)
        note.lineIndex = int(note_payload.get("lineIndex", 0) or 0)
        note.noteLineLayer = int(note_payload.get("noteLineLayer", 0) or 0)
        note.colorType = int(note_payload.get("colorType", 0) or 0)
        note.cutDirection = int(note_payload.get("cutDirection", 0) or 0)
        note.event_time = float(note_payload.get("event_time", 0.0))
        note.spawn_time = float(note_payload.get("spawn_time", 0.0))
        note.event_type = int(note_payload.get("event_type", 0) or 0)
        note.cut = _make_cut(note_payload.get("cut"))
        note.pre_score = int(note_payload.get("pre_score", 0) or 0)
        note.post_score = int(note_payload.get("post_score", 0) or 0)
        note.acc_score = int(note_payload.get("acc_score", 0) or 0)
        note.score = int(note_payload.get("score", 0) or 0)
        replay.notes.append(note)

    replay.walls = []
    for wall_payload in payload.get("walls", []) or []:
        wall = Wall()
        wall.id = int(wall_payload.get("id", 0) or 0)
        wall.energy = float(wall_payload.get("energy", 0.0))
        wall.time = float(wall_payload.get("time", 0.0))
        wall.spawnTime = float(wall_payload.get("spawnTime", 0.0))
        replay.walls.append(wall)

    replay.heights = []
    for height_payload in payload.get("heights", []) or []:
        height = Height()
        height.height = float(height_payload.get("height", 0.0))
        height.time = float(height_payload.get("time", 0.0))
        replay.heights.append(height)

    replay.pauses = []
    for pause_payload in payload.get("pauses", []) or []:
        pause = Pause()
        pause.duration = int(pause_payload.get("duration", 0) or 0)
        pause.time = float(pause_payload.get("time", 0.0))
        replay.pauses.append(pause)

    controller_payload = payload.get("controller_offsets")
    if controller_payload:
        offsets = ControllerOffsets()
        offsets.left = _make_vr_object(controller_payload.get("left", {}) or {})
        offsets.right = _make_vr_object(controller_payload.get("right", {}) or {})
        replay.controller_offsets = offsets
    else:
        replay.controller_offsets = []

    import base64

    replay.user_data = []
    for item_payload in payload.get("user_data", []) or []:
        item = UserData()
        item.key = str(item_payload.get("key", ""))
        item.bytes = base64.b64decode(item_payload.get("bytes_base64", "").encode("ascii"))
        replay.user_data.append(item)

    return replay


def load_bsor(replay_path, backend=None):
    backend = _normalize_backend(backend or os.environ.get("CYBERNOODLES_BSOR_BACKEND", "auto"))
    errors = {}
    if backend == "python":
        ordered_backends = ["python"]
    elif backend == "rust":
        ordered_backends = ["rust"]
    else:
        ordered_backends = ["rust", "python"]

    for candidate in ordered_backends:
        try:
            if candidate == "python":
                with open(replay_path, "rb") as replay_file:
                    return make_bsor(replay_file)
            payload = json.loads(
                _run_bsor_tools(
                    ["dump-json", os.path.abspath(replay_path)],
                    auto_build=(backend == "rust" or "python" in errors),
                )
            )
            return payload_to_bsor(payload)
        except BaseException as exc:
            errors[candidate] = exc

    raise RuntimeError(
        f"Unable to parse BSOR {replay_path} with either backend. "
        f"python={errors.get('python')} | rust={errors.get('rust')}"
    )


def parse_dataset_view_via_rust(replay_path, auto_build=False):
    payload = json.loads(
        _run_bsor_tools(
            ["dump-dataset-json", os.path.abspath(replay_path)],
            auto_build=auto_build,
        )
    )
    return payload.get("frames", []), payload.get("meta", {})


def build_bc_dataset_via_rust(
    replay_dir,
    maps_dir,
    output_dir,
    selected_scores_path,
    *,
    workers=1,
    top_selected=None,
    manifest_save_every=32,
    max_pending_writes=16,
    gc_every=16,
    status_every=25,
):
    args = [
        "build-bc-dataset",
        "--replay-dir",
        os.path.abspath(replay_dir),
        "--maps-dir",
        os.path.abspath(maps_dir),
        "--output-dir",
        os.path.abspath(output_dir),
        "--selected-scores",
        os.path.abspath(selected_scores_path),
        "--workers",
        str(max(1, int(workers))),
        "--manifest-save-every",
        str(max(1, int(manifest_save_every))),
        "--max-pending-writes",
        str(max(1, int(max_pending_writes))),
        "--gc-every",
        str(max(0, int(gc_every))),
        "--status-every",
        str(max(1, int(status_every))),
    ]
    if top_selected is not None:
        args.extend(["--top-selected", str(max(1, int(top_selected)))])
    return _run_bsor_tools(args, auto_build=True, stream_output=True)


def write_bsor(replay, output_path, backend=None):
    backend = _normalize_backend(backend or os.environ.get("CYBERNOODLES_BSOR_WRITE_BACKEND", "python"), default="python")
    if backend != "rust":
        try:
            with open(output_path, "wb") as replay_file:
                replay.write(replay_file)
            return
        except Exception:
            if backend == "python":
                raise

    payload = json.dumps(bsor_to_payload(replay), separators=(",", ":"))
    _run_bsor_tools(
        ["write-json", "--output", os.path.abspath(output_path)],
        stdin_text=payload,
        auto_build=True,
    )


def _python_validation_summary(replay):
    parsed_frames = [
        (
            float(frame.left_hand.x),
            float(frame.left_hand.y),
            float(frame.left_hand.z),
            float(frame.right_hand.x),
            float(frame.right_hand.y),
            float(frame.right_hand.z),
        )
        for frame in getattr(replay, "frames", [])
    ]
    if parsed_frames:
        left_points = [frame[:3] for frame in parsed_frames]
        right_points = [frame[3:] for frame in parsed_frames]
        left_span = math.sqrt(
            sum(
                (max(axis_values) - min(axis_values)) ** 2
                for axis_values in zip(*left_points)
            )
        )
        right_span = math.sqrt(
            sum(
                (max(axis_values) - min(axis_values)) ** 2
                for axis_values in zip(*right_points)
            )
        )
    else:
        left_span = 0.0
        right_span = 0.0

    info = getattr(replay, "info", None)
    return {
        "frame_count": len(getattr(replay, "frames", []) or []),
        "note_count": len(getattr(replay, "notes", []) or []),
        "wall_count": len(getattr(replay, "walls", []) or []),
        "pause_count": len(getattr(replay, "pauses", []) or []),
        "user_data_count": len(getattr(replay, "user_data", []) or []),
        "left_span": float(left_span),
        "right_span": float(right_span),
        "song_hash": str(getattr(info, "songHash", "") if info else ""),
        "difficulty": str(getattr(info, "difficulty", "") if info else ""),
        "mode": str(getattr(info, "mode", "") if info else ""),
    }


def validate_bsor(replay_path, backend=None):
    backend = _normalize_backend(backend or os.environ.get("CYBERNOODLES_BSOR_VALIDATE_BACKEND", "auto"))
    rust_error = None
    rust_available = False

    if backend != "python":
        rust_available = bsor_tools_available(auto_build=(backend == "rust"))
        if backend == "rust" and not rust_available:
            raise RuntimeError(
                "Rust bsor_tools binary is not available. Build it with "
                f"`cargo build --release --manifest-path \"{BSOR_TOOLS_MANIFEST}\"`."
            )

    if backend != "python" and rust_available:
        try:
            summary = json.loads(
                _run_bsor_tools(
                    ["validate", os.path.abspath(replay_path)],
                    auto_build=(backend == "rust"),
                )
            )
            if isinstance(summary, dict):
                summary.setdefault("validation_backend", "rust")
                summary.setdefault("rust_validation_ok", True)
            return summary
        except Exception as exc:
            rust_error = exc
            if backend == "rust":
                raise

    replay = load_bsor(replay_path, backend="python")
    summary = _python_validation_summary(replay)
    summary["validation_backend"] = "python"
    if rust_error is not None:
        summary["rust_validation_error"] = str(rust_error)
        summary["rust_validation_ok"] = False
    elif backend == "auto" and not rust_available:
        summary["rust_validation_skipped"] = True
        summary["rust_validation_skip_reason"] = "bsor_tools_unavailable"
    return summary


def audit_replays_via_rust(replay_dir, *, limit=0, check="both", strict=False, auto_build=False):
    with NamedTemporaryFile("w+", suffix=".json", delete=False) as tmp:
        json_path = tmp.name
    try:
        _run_bsor_tools(
            [
                "audit",
                "--replay-dir",
                os.path.abspath(replay_dir),
                "--limit",
                str(max(0, int(limit))),
                "--check",
                str(check),
                "--json-out",
                json_path,
            ],
            auto_build=auto_build,
        )
        with open(json_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    finally:
        try:
            os.remove(json_path)
        except OSError:
            pass
