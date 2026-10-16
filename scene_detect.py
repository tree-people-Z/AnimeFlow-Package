"""Neural shot-boundary detection for AnimeFlow.

TransNetV2 (Soucek & Lokoc, 2020) classifies every frame as "inside a shot" or
"on a transition".  Compared with FFmpeg's scene-score threshold it is far more
reliable on animation, where a hard cut between two similarly-lit cels can slip
under a pixel-difference threshold while a fast pan can falsely trip one.

The network wants 27x48 RGB frames, so the video is decoded straight to that
size by FFmpeg as raw RGB instead of decoding full-resolution PNGs first.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
# The packaged embeddable Python takes sys.path from python313._pth and does not
# add the running script's directory, so make the app root explicit before
# importing the vendored TransNetV2 under vendor/.
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

TRANSNET_WEIGHTS = APP_DIR / "models" / "transnetv2" / "transnetv2-pytorch-weights.pth"

_HEIGHT, _WIDTH = 27, 48
_WINDOW = 100      # the network's fixed temporal input length
_HALF = 50         # 50% overlap between windows
_PAD = 25          # frames trimmed from each window edge

_WRAPPER_PREFIXES = ("_orig_mod.", "module.", "model.")


def _strip_wrappers(state: dict) -> dict:
    cleaned = {}
    for key, value in state.items():
        changed = True
        while changed:
            changed = False
            for prefix in _WRAPPER_PREFIXES:
                if key.startswith(prefix):
                    key = key[len(prefix):]
                    changed = True
        cleaned[key] = value
    return cleaned


def _load_model(weights: Path, device: str):
    import torch

    from vendor.transnetv2_pytorch import TransNetV2

    state = torch.load(weights, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model = TransNetV2()
    model.load_state_dict(_strip_wrappers(state), strict=True)
    model.eval().to(device)
    return model


def _decode_small_frames(ffmpeg: str, video: str, timeout: int = 3600):
    """Decode the video straight to 27x48 RGB frames."""
    import numpy as np

    command = [ffmpeg, "-v", "error", "-i", video, "-vf", f"scale={_WIDTH}:{_HEIGHT}",
               "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    result = subprocess.run(command, capture_output=True, timeout=timeout)
    if result.returncode != 0:
        raise RuntimeError("FFmpeg 解码失败：" + (result.stderr or b"")[-400:].decode("utf-8", "replace"))
    raw = result.stdout
    frame_bytes = _HEIGHT * _WIDTH * 3
    count = len(raw) // frame_bytes
    if count == 0:
        raise RuntimeError("视频中没有可解码的帧。")
    return np.frombuffer(raw[:count * frame_bytes], dtype=np.uint8).reshape(count, _HEIGHT, _WIDTH, 3)


def _predict(model, frames, device, progress=None):
    import numpy as np
    import torch

    remainder = len(frames) % _HALF
    tail = _PAD + _HALF - (remainder if remainder else _HALF)
    padded = np.concatenate([
        np.repeat(frames[:1], _PAD, axis=0),
        frames,
        np.repeat(frames[-1:], tail, axis=0),
    ], axis=0)

    steps = max(1, (len(padded) - _WINDOW) // _HALF + 1)
    single_parts, many_parts = [], []
    pointer = 0
    step = 0
    while pointer + _WINDOW <= len(padded):
        batch = torch.from_numpy(np.ascontiguousarray(padded[pointer:pointer + _WINDOW])).unsqueeze(0).to(device)
        with torch.inference_mode():
            single, extra = model(batch)
        single_parts.append(torch.sigmoid(single).float().cpu().numpy()[0, :, 0][_PAD:_PAD + _HALF])
        many_parts.append(torch.sigmoid(extra["many_hot"]).float().cpu().numpy()[0, :, 0][_PAD:_PAD + _HALF])
        pointer += _HALF
        step += 1
        if progress is not None:
            progress(0.2 + 0.6 * step / steps, desc=f"TransNetV2 分析 {min(pointer, len(frames))}/{len(frames)} 帧")

    single = np.concatenate(single_parts)[:len(frames)]
    many = np.concatenate(many_parts)[:len(frames)]
    return single, many


def predictions_to_scenes(predictions, threshold: float = 0.5):
    """Port of the reference `predictions_to_scenes` from the TransNetV2 repo."""
    import numpy as np

    binary = (predictions > threshold).astype(np.uint8)
    scenes = []
    t, t_previous, start = -1, 0, 0
    for index, t in enumerate(binary):
        if t_previous == 1 and t == 0:
            start = index
        if t_previous == 0 and t == 1 and index != 0:
            scenes.append([start, index])
        t_previous = t
    if t == 0:
        scenes.append([start, len(predictions)])
    if t == 1:
        scenes.append([start, len(predictions)])
    return np.array(scenes, dtype=np.int32)


def detect(ffmpeg: str, video: str, threshold: float = 0.5, progress=None):
    """Return (cuts, scenes) as frame-index pairs plus the raw predictions."""
    import torch

    if not TRANSNET_WEIGHTS.is_file():
        raise RuntimeError(f"未找到 TransNetV2 权重：{TRANSNET_WEIGHTS}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if progress is not None:
        progress(0.05, desc="解码视频（27×48 低分辨率）")
    frames = _decode_small_frames(ffmpeg, video)

    if progress is not None:
        progress(0.15, desc="加载 TransNetV2")
    model = _load_model(TRANSNET_WEIGHTS, device)

    single, many = _predict(model, frames, device, progress)
    scenes = predictions_to_scenes(many, threshold)
    # predictions_to_scenes leaves the transition frame out of *both* scenes
    # (e.g. [0,7) and [8,16) around a cut at frame 8), so the first frame of
    # each new shot is the start index of the following scene.
    cuts = [int(start) for start, _ in scenes[1:]]
    return cuts, scenes, single, many, len(frames)
