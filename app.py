from __future__ import annotations

import json
import contextlib
import gc
import os
import queue
import shutil
import subprocess
import tempfile
import time
import zipfile
import re
import sys
import threading
from functools import lru_cache, wraps
from pathlib import Path

import gradio as gr


APP_DIR = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent
# The packaged embeddable Python takes sys.path from python313._pth, and in that
# mode the running script's own directory is not added automatically - so put the
# app root on the path explicitly, or sibling modules and vendor/ cannot import.
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))
# All third-party packages (torch, gradio, timm, torchvision, kornia, ...) live
# in the embedded interpreter's own site-packages, so no extra path setup is
# needed when running from the package.  A development virtualenv that lacks
# timm/torchvision/kornia can point at runtime/python instead.
WORK_DIR = APP_DIR / "work"
OUTPUT_DIR = APP_DIR / "outputs"
WORK_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)


def find_binary(name: str, env_name: str) -> str | None:
    configured = os.environ.get(env_name)
    if configured and Path(configured).exists():
        return configured
    bundled = APP_DIR / "bin" / name
    if bundled.exists():
        return str(bundled)
    return shutil.which(name)


FFMPEG = find_binary("ffmpeg", "FFMPEG_PATH")
REALCUGAN = str(APP_DIR / "bin" / "realcugan" / "realcugan-ncnn-vulkan.exe")
# Two Real-CUGAN weight sets ship with the tool.  "Pro" gives noticeably
# cleaner anime line art (the upstream changelog documents fixed ringing,
# reduced texture smearing and no noise amplification), but the author never
# released 4x Pro weights, so 4x has to use the "se" set.
REALCUGAN_MODELS_SE = APP_DIR / "bin" / "realcugan" / "models-se"
REALCUGAN_MODELS_PRO = APP_DIR / "models" / "realcugan-anime" / "models-pro"
FFPROBE = find_binary("ffprobe", "FFPROBE_PATH")
_bundled_models = APP_DIR / "models"
_legacy_models = Path(r"C:\Users\tree-\AppData\Roaming\OnlyScripts\runtime\models")
MODEL_DIR = Path(os.environ.get("ONLYSCRIPTS_MODEL_DIR", str(_bundled_models if _bundled_models.exists() else _legacy_models)))
DEPTH_LARGE_MODEL = MODEL_DIR / "depth-anything-v2-large-hf"


def _batch_setting(name: str, default: int, maximum: int) -> int:
    """Read a safe batch default; the video UI can override it per task."""
    try:
        value = int(os.environ.get(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(1, min(maximum, value))


# Batch size is deliberately conservative for the packaged 8 GB target GPU.
# Depth Anything V2 Large uses a native 518px input, but end-to-end tests show
# batch 1 and 2 are effectively the same speed once warmed.  Keep the smaller
# batch as the default to minimize VRAM use; the video slider can opt into a
# larger shape when it proves useful on another card.
DEPTH_BATCH_MAX = 4
CUTOUT_BATCH_MAX = 4
DEPTH_BATCH_SIZE = _batch_setting("ANIMEFLOW_DEPTH_BATCH_SIZE", 1, DEPTH_BATCH_MAX)
CUTOUT_BATCH_SIZE = _batch_setting("ANIMEFLOW_CUTOUT_BATCH_SIZE", 1, CUTOUT_BATCH_MAX)

# Real-CUGAN picks its weights as <model-dir>/up{scale}x-{suffix}.
_REALCUGAN_NOISE_SUFFIX = {
    -1: "conservative", 0: "no-denoise", 1: "denoise1x", 2: "denoise2x", 3: "denoise3x",
}

# Real-CUGAN is the only anime upscaler kept: it is the best-rated option for
# cel animation and Real-ESRGAN AnimeVideoV3 measured worse (over-sharpening,
# art-style drift), so it was removed rather than offered as a second choice.
# The scale is a parameter, not a different model.  Pro supplies 2x/3x; 4x only
# ever shipped in the "se" set because the author was unhappy with Pro 4x.
UPSCALE_PRESETS = {
    "Real-CUGAN Pro 2×（动漫推荐）": ("pro", 2),
    "Real-CUGAN Pro 3×": ("pro", 3),
    "Real-CUGAN 4×（SE 权重）": ("se", 4),
}
UPSCALE_CHOICES = list(UPSCALE_PRESETS)

DENOISE_CHOICES = {
    "保守 -1（保留细节 · 推荐）": -1,
    "不降噪 0": 0,
    "强力降噪 3（片源噪点多）": 3,
}
DEFAULT_DENOISE = "保守 -1（保留细节 · 推荐）"


# CUDA model ownership is coordinated here instead of letting each backend
# keep an unrelated lru_cache alive.  A backend stays warm while the same
# operation is being used, but switching operations releases the previous
# backend before loading another one.
_MODEL_LOCK = threading.RLock()
_ACTIVE_BACKEND = None


def _clear_cuda_cache():
    """Drop Python references and return unused PyTorch blocks to the driver."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        # Cleanup must never turn a completed media task into an error.
        pass


def _release_backend(backend: str | None):
    """Release one backend's cached resources in the current process."""
    if backend == "relight":
        _marigold_pipeline.cache_clear()
    elif backend == "depth":
        _depth_large_pipeline.cache_clear()
    elif backend == "vfi":
        try:
            import vfi_plus
            vfi_plus.release()
        except Exception:
            pass
    elif backend == "cutout":
        try:
            import gpu_cutout
            gpu_cutout.release()
        except Exception:
            pass


def _release_all_backends(except_backend: str | None = None):
    """Release every cached CUDA backend except the one being activated."""
    for backend in ("relight", "depth", "vfi", "cutout"):
        if backend != except_backend:
            _release_backend(backend)
    _clear_cuda_cache()


@contextlib.contextmanager
def _model_scope(backend: str):
    """Serialize GPU tasks and switch to an exclusive active model backend."""
    global _ACTIVE_BACKEND
    with _MODEL_LOCK:
        if _ACTIVE_BACKEND != backend:
            _release_all_backends(except_backend=backend)
            _ACTIVE_BACKEND = backend
        yield


def _exclusive_backend(backend: str):
    """Decorate a public task so its model remains exclusive for its run."""
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with _model_scope(backend):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def _realcugan_model_dir(model_set: str) -> Path:
    return REALCUGAN_MODELS_PRO if model_set == "pro" else REALCUGAN_MODELS_SE


def _realcugan_weights(model_set: str, scale: int, noise: int) -> Path:
    """Return the weight directory, proving the requested variant is present."""
    suffix = _REALCUGAN_NOISE_SUFFIX.get(noise)
    if suffix is None:
        raise gr.Error(f"不支持的降噪强度：{noise}")
    directory = _realcugan_model_dir(model_set)
    weight = directory / f"up{scale}x-{suffix}.bin"
    if not weight.is_file():
        available = sorted(p.name for p in directory.glob("up*.bin"))
        raise gr.Error(
            f"{'Pro' if model_set == 'pro' else 'SE'} 模型集没有 {scale}× 降噪 {noise} 的权重（{weight.name}）。"
            f"可用：{', '.join(available) or '无'}"
        )
    return directory


def _realcugan_label(model_set: str, scale: int) -> str:
    return f"Real-CUGAN {'Pro ' if model_set == 'pro' else ''}{scale}×"


def run(command: list[str], **kwargs) -> subprocess.CompletedProcess[str]:
    kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    kwargs.setdefault("encoding", "utf-8")
    kwargs.setdefault("errors", "replace")
    return subprocess.run(command, **kwargs)


def run_ffmpeg_with_frame_progress(command: list[str], total: int | None,
                                   progress, start: float, end: float,
                                   label: str) -> subprocess.CompletedProcess[str]:
    """Run FFmpeg while forwarding its decoded/encoded frame counter to UI."""
    import tempfile

    tracked = list(command) + ["-progress", "pipe:1", "-nostats"]
    # Keep stderr on a seekable temporary file. Reading stdout synchronously is
    # what makes live progress possible; a PIPE for stderr could fill its OS
    # buffer while FFmpeg is writing image-sequence warnings and deadlock the
    # process before it emits the final progress event.
    stderr_file = tempfile.TemporaryFile()
    process = subprocess.Popen(
        tracked, stdout=subprocess.PIPE, stderr=stderr_file,
        text=True, encoding="utf-8", errors="replace",
    )
    last_frame = -1
    if process.stdout is not None:
        for line in process.stdout:
            if not line.startswith("frame="):
                continue
            try:
                current = int(line.split("=", 1)[1].strip())
            except ValueError:
                continue
            if current == last_frame:
                continue
            last_frame = current
            if progress is not None and total:
                progress(start + (end - start) * min(current / total, 1.0),
                         desc=f"{label} · {current}/{total} 帧")
    returncode = process.wait()
    stderr_file.seek(0)
    stderr = stderr_file.read().decode("utf-8", "replace")
    stderr_file.close()
    return subprocess.CompletedProcess(tracked, returncode, "", stderr)


def backend_status() -> str:
    ffmpeg_state = "已找到" if FFMPEG else "未找到"
    ffprobe_state = "已找到" if FFPROBE else "未找到"
    try:
        import vfi_plus
        interp_state = "已配置" if vfi_plus.available() else "未配置"
    except Exception:
        interp_state = "未配置"
    try:
        import scene_detect
        scene_state = "已配置" if scene_detect.TRANSNET_WEIGHTS.is_file() else "未配置"
    except Exception:
        scene_state = "未配置"
    model_state = "已找到" if MODEL_DIR.exists() else "未找到"
    try:
        import torch
        device_state = f"CUDA · {torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU（补帧/抠图不可用）"
    except Exception:
        device_state = "CPU（补帧/抠图不可用）"
    realcugan_state = "已找到" if Path(REALCUGAN).exists() and REALCUGAN_MODELS_SE.exists() else "未找到"
    return (f"FFmpeg  {ffmpeg_state}　·　FFprobe  {ffprobe_state}　·　"
            f"设备 {device_state}　·　补帧（GMFSS + RIFE 4.26-heavy）{interp_state}　·　"
            f"镜头检测（TransNetV2）{scene_state}　·　Real-CUGAN {realcugan_state}　·　模型目录 {model_state}")


def model_catalog() -> str:
    depth_path = DEPTH_LARGE_MODEL / "model.safetensors"
    depth_label = "Depth Anything V2 Large"
    weights = APP_DIR / "vendor" / "drba" / "weights" / "train_log_gmfss_union"
    entries = [
        ("动画补帧（默认）", weights / "rife.pkl", "GMFSS Fortuna · AnimeRun + RIFE 4.26-heavy · CUDA"),
        ("动画超分（2×/3×）", REALCUGAN_MODELS_PRO / "up2x-conservative.bin", "Real-CUGAN Pro · Vulkan"),
        ("动画超分（4×）", REALCUGAN_MODELS_SE / "up4x-conservative.bin", "Real-CUGAN SE · Vulkan"),
        ("动漫抠图", MODEL_DIR / "toonout-pytorch" / "birefnet_finetuned_toonout.pth", "ToonOut · BiRefNet · CUDA"),
        ("镜头检测", MODEL_DIR / "transnetv2" / "transnetv2-pytorch-weights.pth", "TransNetV2 · CUDA"),
        ("高质量深度估计", depth_path, depth_label),
        ("高质量重打光", MODEL_DIR / "only-shadow-marigold-normals-v1-1" / "unet" / "diffusion_pytorch_model.fp16.safetensors", "Marigold Normals · CUDA"),
    ]
    lines = ["**实际模型**"]
    lines.extend(f"- **{name}**：{model}　—　{'已配置' if path.exists() else '缺失'}" for name, path, model in entries)
    lines.append("")
    lines.append("所有模型均从整合包本地目录加载；缺失时会直接报错，不会静默换成其它模型。")
    return "\n".join(lines)


def media_info(path: str) -> dict:
    if not FFPROBE:
        return {}
    result = run([FFPROBE, "-v", "error", "-show_streams", "-show_format", "-of", "json", path])
    if result.returncode != 0:
        return {}
    try:
        return json.loads(result.stdout)
    except json.JSONDecodeError:
        return {}


def frame_count(path: str) -> int | None:
    if not FFPROBE:
        return None
    result = run([FFPROBE, "-v", "error", "-select_streams", "v:0", "-count_frames",
                  "-show_entries", "stream=nb_read_frames", "-of", "default=nw=1:nk=1", path])
    try:
        return int(result.stdout.strip())
    except ValueError:
        return None


def fps(path: str) -> float:
    info = media_info(path)
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            value = stream.get("avg_frame_rate", "0/1")
            try:
                n, d = value.split("/")
                return float(n) / float(d)
            except (ValueError, ZeroDivisionError):
                pass
    return 24.0


def estimated_frame_count(path: str) -> int | None:
    """Read a frame total without decoding the entire video a second time."""
    info = media_info(path)
    for stream in info.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        try:
            count = int(stream.get("nb_frames", 0))
            if count > 0:
                return count
        except (TypeError, ValueError):
            pass
        try:
            duration = float(stream.get("duration", 0))
            rate = stream.get("avg_frame_rate", "0/1")
            numerator, denominator = rate.split("/")
            value = duration * float(numerator) / float(denominator)
            if value > 0:
                return max(1, round(value))
        except (AttributeError, TypeError, ValueError, ZeroDivisionError):
            pass
    return None


def video_shape(path: str) -> tuple[int, int] | None:
    """Return the source video width and height from its container metadata."""
    info = media_info(path)
    for stream in info.get("streams", []):
        if stream.get("codec_type") != "video":
            continue
        try:
            return int(stream["width"]), int(stream["height"])
        except (KeyError, TypeError, ValueError):
            return None
    return None


def raw_video_frames(path: str, width: int, height: int):
    """Yield RGB24 frames directly from FFmpeg without intermediate image files."""
    import numpy as np

    frame_bytes = width * height * 3
    process = subprocess.Popen(
        [FFMPEG, "-v", "error", "-i", path, "-map", "0:v:0", "-vsync", "0",
         "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    try:
        while True:
            chunk = process.stdout.read(frame_bytes)
            if not chunk:
                break
            while len(chunk) < frame_bytes:
                tail = process.stdout.read(frame_bytes - len(chunk))
                if not tail:
                    raise RuntimeError("FFmpeg 原始帧管道提前结束。")
                chunk += tail
            yield np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3)
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()
    if process.returncode != 0:
        raise RuntimeError("FFmpeg 原始帧解码失败。")


def raw_video_sample_frames(path: str, width: int, height: int, step: int):
    """Yield every *step*th RGB frame without decoding the other frames.

    The video-depth pass only needs a small, evenly spaced sample to establish
    one global percentile range.  FFmpeg's ``select`` filter keeps the same
    frame positions as the old Python-side modulo check while avoiding a full
    first decode of the clip.
    """
    import numpy as np

    step = max(1, int(step))
    frame_bytes = width * height * 3
    process = subprocess.Popen(
        [FFMPEG, "-v", "error", "-i", path, "-map", "0:v:0", "-vf",
         f"select=not(mod(n\\,{step}))", "-vsync", "0", "-f", "rawvideo",
         "-pix_fmt", "rgb24", "-"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )
    try:
        while True:
            chunk = process.stdout.read(frame_bytes)
            if not chunk:
                break
            while len(chunk) < frame_bytes:
                tail = process.stdout.read(frame_bytes - len(chunk))
                if not tail:
                    raise RuntimeError("FFmpeg 采样帧管道提前结束。")
                chunk += tail
            yield np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3)
    finally:
        process.stdout.close()
        if process.poll() is None:
            process.terminate()
        process.wait()
    if process.returncode != 0:
        raise RuntimeError("FFmpeg 采样帧解码失败。")


def _depth_batch_producer(path: str, width: int, height: int, batch_size: int):
    """Decode and convert bounded depth batches ahead of GPU inference."""
    from PIL import Image

    batches = queue.Queue(maxsize=2)
    stop_event = threading.Event()
    sentinel = object()
    errors = []

    def put(item):
        while not stop_event.is_set():
            try:
                batches.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def produce():
        try:
            prepared = []
            for frame in raw_video_frames(path, width, height):
                if stop_event.is_set():
                    return
                prepared.append(Image.fromarray(frame, mode="RGB"))
                if len(prepared) >= batch_size:
                    if not put(prepared):
                        return
                    prepared = []
            if prepared:
                put(prepared)
        except Exception as exc:
            errors.append(exc)
        finally:
            put(sentinel)

    thread = threading.Thread(target=produce, name="depth-decode", daemon=True)
    thread.start()
    return batches, thread, stop_event, sentinel, errors


def _prepare_cutout_frame(frame):
    """Resize/normalise one RGB frame before it crosses the GPU worker IPC."""
    import numpy as np
    from PIL import Image

    image = Image.fromarray(frame, mode="RGB")
    resized = np.asarray(
        image.resize((1024, 1024), Image.Resampling.BICUBIC), dtype=np.float32
    ) / 255.0
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    normalized = ((resized - mean) / std).transpose(2, 0, 1)
    # Keep float32 here to match the old worker path exactly; the final FP16
    # cast still happens during the asynchronous pinned H2D copy.
    return np.ascontiguousarray(normalized, dtype=np.float32).tobytes()


def _cutout_batch_producer(source, width, height, total_hint, batch_size,
                           output_dir=None, stop_event=None):
    """Decode and prepare bounded batches ahead of the GPU worker.

    The queue is intentionally small: two batches overlap FFmpeg/PIL work with
    GPU inference without retaining an entire video in memory.
    """
    batches = queue.Queue(maxsize=2)
    stop_event = stop_event or threading.Event()
    sentinel = object()

    def enqueue(item):
        while not stop_event.is_set():
            try:
                batches.put(item, timeout=0.2)
                return True
            except queue.Full:
                continue
        return False

    def produce():
        try:
            frames = []
            prepared = []
            targets = []
            first_index = 1
            for index, frame in enumerate(raw_video_frames(source, width, height), 1):
                if stop_event.is_set():
                    break
                normalized = _prepare_cutout_frame(frame)
                frames.append(frame.tobytes())
                prepared.append(normalized)
                targets.append(
                    output_dir / f"frame_{index:06d}.png"
                    if output_dir is not None else ""
                )
                if len(frames) >= batch_size or index == total_hint:
                    if not enqueue((first_index, index, frames, prepared, targets)):
                        return
                    frames, prepared, targets = [], [], []
                    first_index = index + 1
            if frames and not stop_event.is_set():
                enqueue((first_index, first_index + len(frames) - 1,
                         frames, prepared, targets))
        except Exception as exc:
            enqueue((sentinel, exc))
        finally:
            if not stop_event.is_set():
                enqueue(sentinel)

    thread = threading.Thread(target=produce, name="cutout-prefetch", daemon=True)
    thread.start()
    return batches, thread, stop_event, sentinel


class _CutoutRGBAWriter:
    """Write RGBA batches to FFmpeg while the next batch is inferred."""

    def __init__(self, encoder, width: int, height: int):
        self.encoder = encoder
        self.width = int(width)
        self.height = int(height)
        self.queue = queue.Queue(maxsize=2)
        self.error = None
        self.stop = object()
        self.thread = threading.Thread(target=self._run, name="cutout-rgba-writer", daemon=True)
        self.thread.start()

    def _run(self):
        import numpy as np
        from PIL import Image

        try:
            while True:
                item = self.queue.get()
                try:
                    if item is self.stop:
                        return
                    frames, alpha_bytes = item
                    alpha = np.frombuffer(alpha_bytes, dtype=np.float16).reshape(
                        len(frames), 1024, 1024
                    )
                    for raw, mask_array in zip(frames, alpha):
                        rgb = np.frombuffer(raw, dtype=np.uint8).reshape(
                            self.height, self.width, 3
                        )
                        # Keep the same Lanczos alpha resize as the old path,
                        # but concatenate the final RGBA frame in NumPy.
                        mask = Image.fromarray(
                            np.clip(mask_array * 255.0, 0, 255).astype(np.uint8), mode="L"
                        ).resize((self.width, self.height), Image.Resampling.LANCZOS)
                        rgba = np.empty((self.height, self.width, 4), dtype=np.uint8)
                        rgba[:, :, :3] = rgb
                        rgba[:, :, 3] = np.asarray(mask, dtype=np.uint8)
                        self.encoder.stdin.write(rgba.tobytes())
                finally:
                    self.queue.task_done()
        except Exception as exc:
            self.error = exc
            # Mark queued items done so close()/abort() cannot deadlock after
            # a broken FFmpeg pipe or a disk write failure.
            while True:
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except queue.Empty:
                    break

    def put(self, frames, alpha_bytes):
        while True:
            if self.error is not None:
                raise RuntimeError(f"Alpha 视频写入失败：{self.error}") from self.error
            try:
                self.queue.put((frames, alpha_bytes), timeout=0.2)
                return
            except queue.Full:
                continue

    def close(self):
        while self.queue.unfinished_tasks:
            if self.error is not None:
                raise RuntimeError(f"Alpha 视频写入失败：{self.error}") from self.error
            time.sleep(0.05)
        if self.error is not None:
            raise RuntimeError(f"Alpha 视频写入失败：{self.error}") from self.error
        self.queue.put(self.stop)
        self.thread.join(timeout=10)
        if self.error is not None:
            raise RuntimeError(f"Alpha 视频写入失败：{self.error}") from self.error

    def abort(self):
        """Stop the writer without waiting on queued output after a failure."""
        try:
            while True:
                self.queue.get_nowait()
                self.queue.task_done()
        except queue.Empty:
            pass
        try:
            self.queue.put_nowait(self.stop)
        except queue.Full:
            pass
        self.thread.join(timeout=3)


class _FramePipeWriter:
    """Write ordered raw frames in a bounded thread behind model inference."""

    def __init__(self, encoder, convert=None, name="raw-frame-writer"):
        self.encoder = encoder
        self.convert = convert
        self.queue = queue.Queue(maxsize=2)
        self.error = None
        self.stop = object()
        self.thread = threading.Thread(target=self._run, name=name, daemon=True)
        self.thread.start()

    def _run(self):
        try:
            while True:
                item = self.queue.get()
                try:
                    if item is self.stop:
                        return
                    payload = self.convert(item) if self.convert else item
                    if not isinstance(payload, (bytes, bytearray, memoryview)):
                        import numpy as np
                        payload = np.ascontiguousarray(payload).tobytes()
                    self.encoder.stdin.write(payload)
                finally:
                    self.queue.task_done()
        except Exception as exc:
            self.error = exc
            while True:
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except queue.Empty:
                    break

    def put(self, frame):
        while True:
            if self.error is not None:
                raise RuntimeError(f"视频帧写入失败：{self.error}") from self.error
            try:
                self.queue.put(frame, timeout=0.2)
                return
            except queue.Full:
                continue

    def close(self):
        while self.queue.unfinished_tasks:
            if self.error is not None:
                raise RuntimeError(f"视频帧写入失败：{self.error}") from self.error
            time.sleep(0.05)
        if self.error is not None:
            raise RuntimeError(f"视频帧写入失败：{self.error}") from self.error
        self.queue.put(self.stop)
        self.thread.join(timeout=10)
        if self.error is not None:
            raise RuntimeError(f"视频帧写入失败：{self.error}") from self.error

    def abort(self):
        try:
            while True:
                self.queue.get_nowait()
                self.queue.task_done()
        except queue.Empty:
            pass
        try:
            self.queue.put_nowait(self.stop)
        except queue.Full:
            pass
        self.thread.join(timeout=3)


def video_bitrate(path: str) -> int | None:
    """Return the source video bitrate in bits/s when the container reports it."""
    info = media_info(path)
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            try:
                value = int(float(stream.get("bit_rate", 0)))
                return value if value > 0 else None
            except (TypeError, ValueError):
                return None
    return None


def interpolation_bitrate(path: str, factor: int) -> int:
    """Keep bitrate from dropping after frame interpolation.

    More generated frames need more bits per second. We target at least the
    original bitrate and normally scale it with the interpolation factor,
    with an 8 Mbps floor for clean animation edges.
    """
    source = video_bitrate(path) or 8_000_000
    return max(8_000_000, int(source * max(1, factor) * 1.15))


def ensure_video(path: str | None) -> str:
    if not path:
        raise gr.Error("请先选择一个视频文件")
    if not FFMPEG:
        raise gr.Error("没有找到 FFmpeg。请安装 FFmpeg，或设置 FFMPEG_PATH 环境变量。")
    return path


def deduplicate(video_path: str | None, sensitivity: float, output_format: str, progress=gr.Progress()):
    source = ensure_video(video_path)
    source_fps = fps(source)
    before = estimated_frame_count(source) or frame_count(source)
    extension = "mkv" if output_format == "无损中间文件（MKV）" else "mp4"
    target = OUTPUT_DIR / f"{Path(source).stem}_dedup.{extension}"
    # mpdecimate detects repeated/near-repeated animation cels.  Re-timing the
    # remaining frames makes them a clean input sequence for interpolation.
    hi = max(24, int(sensitivity * 64))
    lo = max(12, int(sensitivity * 32))
    vf = f"mpdecimate=hi={hi}:lo={lo}:frac=0.1,setpts=N/{source_fps}/TB"
    codec = ["-c:v", "ffv1"] if extension == "mkv" else ["-c:v", "libx264", "-crf", "12", "-preset", "medium"]
    command = [FFMPEG, "-y", "-i", source, "-an", "-vf", vf, "-fps_mode", "vfr", *codec, str(target)]
    progress(0.05, desc=f"动画去重 · 准备处理 {before or '?'} 帧")
    result = run_ffmpeg_with_frame_progress(command, before, progress, 0.05, 0.88, "动画去重")
    if result.returncode != 0:
        raise gr.Error("FFmpeg 处理失败：" + result.stderr[-700:])
    after = frame_count(str(target))
    removed = (before - after) if before is not None and after is not None else None
    detail = f"原始 {before or '?'} 帧 → 保留 {after or '?'} 帧"
    if removed is not None:
        detail += f"，移除 {removed} 帧（{removed / max(before, 1):.1%}）"
    detail += f"\n源帧率 {source_fps:.2f} FPS。建议将结果交给补帧引擎生成 2× 或 4× 帧率。"
    progress(1, desc=f"动画去重完成 · {before or '?'} → {after or '?'} 帧")
    return str(target), detail


def extract_frames(video_path: str | None, every: int, fmt: str, progress=gr.Progress()):
    source = ensure_video(video_path)
    total_source = estimated_frame_count(source) or frame_count(source)
    job = Path(tempfile.mkdtemp(prefix="frames_", dir=WORK_DIR))
    ext = "png" if fmt == "PNG（无损）" else "jpg"
    pattern = str(job / f"frame_%06d.{ext}")
    vf = f"select=not(mod(n\\,{max(1, every)}))"
    command = [FFMPEG, "-y", "-i", source, "-vf", vf, "-vsync", "vfr"]
    if ext == "jpg":
        command += ["-q:v", "2"]
    command += [pattern]
    progress(0.05, desc=f"导出帧 · 准备处理 {total_source or '?'} 帧")
    result = run_ffmpeg_with_frame_progress(command, total_source, progress, 0.05, 0.78, "导出帧")
    if result.returncode != 0:
        raise gr.Error("FFmpeg 抽帧失败：" + result.stderr[-700:])
    archive = OUTPUT_DIR / f"{Path(source).stem}_frames.zip"
    files = sorted(job.glob(f"*.{ext}"))
    progress(0.8, desc=f"导出帧完成 · 正在打包 0/{len(files)} 帧")
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for index, file in enumerate(files, 1):
            zf.write(file, file.name)
            if index == 1 or index % 10 == 0 or index == len(files):
                progress(0.8 + 0.18 * index / max(len(files), 1),
                         desc=f"导出帧 · 正在打包 {index}/{len(files)} 帧")
    duplicate_previews = []
    try:
        from PIL import Image, ImageChops
        previous = None
        for file in files:
            current = Image.open(file).convert("RGB").resize((96, 96))
            if previous is not None:
                diff = ImageChops.difference(previous, current)
                if sum(diff.resize((1, 1)).getpixel((0, 0))) / 3 < 5:
                    duplicate_previews.append((current, file.name))
                    if len(duplicate_previews) >= 24:
                        break
            previous = current
    except Exception:
        duplicate_previews = []
    shutil.rmtree(job, ignore_errors=True)
    progress(1, desc=f"导出帧完成 · 共 {len(files)} 张")
    detail = f"已导出 {len(files)} 张图片。"
    detail += f" 检测到 {len(duplicate_previews)} 张连续重复/近重复帧。" if duplicate_previews else " 未检测到连续重复帧。"
    return str(archive), duplicate_previews, detail


INTERP_ENGINES = ["GMFSS Fortuna + RIFE 4.26-heavy"]


def interpolate(video_path: str | None, prepared_path: str | None, factor: int,
                engine: str = INTERP_ENGINES[0], progress=gr.Progress()):
    source = ensure_video(prepared_path or video_path)
    if engine not in INTERP_ENGINES:
        raise gr.Error(f"未知补帧模型：{engine}")
    return enhanced_video(source, int(factor), progress=progress)


@_exclusive_backend("vfi")
def enhanced_video(source: str, factor: int, progress=gr.Progress()):
    """Interpolate with GMFSS Fortuna union weights plus the RIFE 4.26-heavy IFNet.

    Measured against ground truth on real 1080p anime (two sources x three
    motion profiles), swapping GMFSS's stock rife branch for RIFE 4.26-heavy is
    worth +0.67 dB on average and helps in 4 of 6 cases.  Competing options were
    removed because they measured worse: plain GMFSS (older IFNet), RIFE Anime
    2022 (ncnn, frozen 2022 engine) and DRBA's per-pixel DRM timing (-0.41 dB).
    """
    if factor not in (2, 3, 4):
        raise gr.Error("该补帧引擎支持 2×、3× 或 4×")
    if not torch_cuda_available():
        raise gr.Error("该补帧引擎需要 CUDA 显卡；未检测到 CUDA，未切换到其他模型。")

    import vfi_plus
    if not vfi_plus.available():
        raise gr.Error(f"补帧引擎文件不完整：{vfi_plus.WEIGHTS}")

    import numpy as np

    label = "GMFSS Fortuna + RIFE 4.26-heavy"
    suffix = "gmfss-rife426"

    target = OUTPUT_DIR / f"{Path(source).stem}_{factor}x_{suffix}.mp4"
    source_rate = fps(source)
    info = media_info(source)
    video_stream = next((stream for stream in info.get("streams", [])
                         if stream.get("codec_type") == "video"), {})
    try:
        width = int(video_stream["width"])
        height = int(video_stream["height"])
    except (KeyError, TypeError, ValueError):
        raise gr.Error("无法读取视频分辨率。")
    total_frames = estimated_frame_count(source)

    decoder = None
    encoder = None
    frame_writer = None
    try:
        if total_frames is not None and total_frames < 2:
            raise gr.Error("视频至少需要两帧才能补帧。")

        frame_note = f"源帧约 {total_frames} · 目标帧约 {total_frames * factor}" if total_frames else "源帧总数读取中"
        progress(0.03, desc=f"加载 {label} · CUDA · {frame_note}")

        frame_bytes = width * height * 3
        decoder = subprocess.Popen(
            [FFMPEG, "-v", "error", "-i", source, "-map", "0:v:0", "-vsync", "0",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )

        def source_frames():
            while True:
                chunk = decoder.stdout.read(frame_bytes)
                if not chunk:
                    break
                while len(chunk) < frame_bytes:
                    tail = decoder.stdout.read(frame_bytes - len(chunk))
                    if not tail:
                        raise RuntimeError("FFmpeg 原始帧管道提前结束。")
                    chunk += tail
                yield np.frombuffer(chunk, dtype=np.uint8).reshape(height, width, 3)

        target_rate = source_rate * factor
        target_bitrate = interpolation_bitrate(source, factor)
        encoder = subprocess.Popen(
            [FFMPEG, "-y", "-v", "error",
             "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
             "-framerate", f"{target_rate:.8f}", "-i", "-",
             "-i", source, "-map", "0:v:0", "-map", "1:a?",
             "-c:v", "libx264", "-crf", "10", "-preset", "slow", "-tune", "animation",
             "-b:v", str(target_bitrate), "-maxrate", str(target_bitrate),
             "-bufsize", str(target_bitrate * 2), "-pix_fmt", "yuv420p",
             "-c:a", "copy", "-shortest", "-movflags", "+faststart", str(target)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        frame_writer = _FramePipeWriter(encoder, name="vfi-frame-writer")

        counter = {"n": 0}

        def emit(frame):
            counter["n"] += 1
            frame = np.ascontiguousarray(frame, dtype=np.uint8)
            if frame.shape != (height, width, 3):
                raise RuntimeError(f"模型输出尺寸异常：{frame.shape}，预期 {(height, width, 3)}")
            frame_writer.put(frame)

        started = time.perf_counter()
        vfi_plus.interpolate_stream(source_frames(), int(factor), emit,
                                    progress=progress, total_hint=total_frames or 0)
        if decoder.wait() != 0:
            detail = decoder.stderr.read().decode("utf-8", "replace")[-700:]
            raise gr.Error("FFmpeg 解码失败：" + detail)
        decoder.stdout.close()
        decoder.stderr.close()
        decoder = None

        output_note = f"/{total_frames * factor}" if total_frames else ""
        progress(0.95, desc=f"正在完成编码 · 已写入 {counter['n']}{output_note} 帧")
        frame_writer.close()
        frame_writer = None
        encoder.stdin.close()
        encoder.stdin = None
        if encoder.wait() != 0:
            detail = encoder.stderr.read().decode("utf-8", "replace")[-700:]
            raise gr.Error("视频编码失败：" + detail)
        elapsed = time.perf_counter() - started

        if counter["n"] < 2:
            raise gr.Error("没有生成足够的画面。")

        progress(1, desc=f"完成 · {label}")
        return str(target), (
            f"实际模型：{label} · CUDA。"
            f"{total_frames or '?'} → {counter['n']} 帧，{source_rate:.2f} → {target_rate:.2f} FPS，"
            f"管道补帧与编码 {elapsed / 60:.1f} 分钟（{elapsed / max((total_frames or counter['n']) - 1, 1):.2f} 秒/源帧），"
            f"目标视频码率 {target_bitrate/1_000_000:.1f} Mbps。"
        )
    finally:
        if frame_writer is not None:
            frame_writer.abort()
        if decoder is not None:
            if decoder.stdout:
                decoder.stdout.close()
            if decoder.poll() is None:
                decoder.terminate()
            try:
                decoder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                decoder.kill()
                decoder.wait()
        if encoder is not None:
            if encoder.stdin:
                encoder.stdin.close()
            if encoder.poll() is None:
                encoder.terminate()
            try:
                encoder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                encoder.kill()
                encoder.wait()


def torch_cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


def _resolve_preset(mode: str):
    preset = UPSCALE_PRESETS.get(mode)
    if preset is None:
        raise gr.Error(f"未知的超分模式：{mode}")
    return preset


@_exclusive_backend("upscale")
def video_enhance(video_path: str | None, mode: str, denoise: str = DEFAULT_DENOISE,
                  tta: bool = False, progress=gr.Progress()):
    """Run the selected anime upscaler."""
    source = ensure_video(video_path)
    model_set, scale = _resolve_preset(mode)
    return realcugan_video(source, scale, model_set, DENOISE_CHOICES.get(denoise, -1), bool(tta), progress)

SCENE_ENGINES = ["TransNetV2"]


def _timecode(seconds: float) -> str:
    hours, rest = divmod(max(seconds, 0.0), 3600)
    minutes, secs = divmod(rest, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{secs:06.3f}"


def _write_scene_report(source: str, engine: str, frame_rate: float, total: int,
                        cuts: list[int], scenes: list[tuple[int, int]], note: str) -> str:
    lines = [
        f"# AnimeFlow 镜头检测 · {engine}",
        f"# 源文件 {Path(source).name}",
        f"# 帧率 {frame_rate:.3f} FPS · 共 {total} 帧",
        f"# 场景 {len(scenes)} 个 · 切镜点 {len(cuts)} 个",
        "",
        "## 切镜点",
    ]
    if cuts:
        lines += [f"{_timecode(c / frame_rate)}  帧 {c}" for c in cuts]
    else:
        lines.append("（未检测到切镜）")
    lines += ["", "## 场景区间"]
    for index, (start, end) in enumerate(scenes):
        lines.append(
            f"[{index:03d}] {_timecode(start / frame_rate)} → {_timecode(end / frame_rate)}"
            f"  帧 {start}-{end - 1}（{end - start} 帧）"
        )
    lines += ["", f"# {note}"]
    report = OUTPUT_DIR / f"{Path(source).stem}_scenes.txt"
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(report)


@_exclusive_backend("scene")
def detect_scenes(video_path: str | None, threshold: float, progress=gr.Progress()):
    """Detect shot boundaries with TransNetV2 and write an editable report.

    FFmpeg's scene-score threshold was dropped: it is a pixel-difference
    heuristic that misses hard cuts between similarly-lit cels and false-fires on
    fast pans, which is exactly what anime does most.
    """
    source = ensure_video(video_path)
    return _scenes_transnet(source, threshold, fps(source), progress)


def _scenes_transnet(source: str, threshold: float, frame_rate: float, progress=gr.Progress()):
    import scene_detect

    try:
        total_hint = estimated_frame_count(source)
        progress(0.03, desc=f"TransNetV2 · 准备分析 {total_hint or '?'} 帧")
        cuts, scenes, _single, _many, total = scene_detect.detect(
            FFMPEG, source, threshold, progress)
    except Exception as exc:
        raise gr.Error(f"TransNetV2 镜头检测失败：{exc}") from exc

    pairs = [(int(s), int(e)) for s, e in scenes]
    note = ("TransNetV2 神经网络逐帧判别，对动画硬切与快速运动都稳定；"
            "阈值是转场概率，0.5 为官方默认，调低更敏感。")
    report = _write_scene_report(source, "TransNetV2", frame_rate, total, cuts, pairs, note)
    progress(1, desc="完成 · TransNetV2")
    return report, (
        f"TransNetV2 检测到 {len(cuts)} 个切镜、{len(pairs)} 个场景，共 {total} 帧"
        f"（{_timecode(total / frame_rate)}）。阈值 {threshold:.2f}。"
    )


def image_stub(image_path: str | None, task: str, mode: str = "Real-CUGAN Pro 2×（动漫推荐）",
               denoise: str = DEFAULT_DENOISE, tta: bool = False):
    if not image_path:
        raise gr.Error("请先选择一张图片")
    if task == "深度图":
        return depth_image(image_path)
    if task == "超分":
        return upscale_image(image_path, mode, denoise, tta)
    if task == "抠图":
        return cutout_image(image_path)
    if task == "重新打光":
        return relight_image(image_path)
    raise gr.Error(f"{task} 模型后端尚未接入。请先在设置中配置本地模型运行时。")


@_exclusive_backend("batch")
def batch_images(image_paths: list[str] | None, task: str, mode: str = "Real-CUGAN Pro 2×（动漫推荐）",
                 denoise: str = DEFAULT_DENOISE, tta: bool = False, progress=gr.Progress()):
    if not image_paths:
        raise gr.Error("请先选择图片")
    produced = []
    errors = []
    for index, path in enumerate(image_paths, 1):
        try:
            output, _ = image_stub(path, task, mode, denoise, tta)
            produced.append(Path(output))
        except Exception as exc:
            errors.append(f"{Path(path).name}: {exc}")
        progress(index / len(image_paths), desc=f"批量处理 {index}/{len(image_paths)}")
    archive = OUTPUT_DIR / "batch_images.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for output in produced:
            zf.write(output, output.name)
    detail = f"完成 {len(produced)}/{len(image_paths)} 张图片。"
    if errors:
        detail += "\n" + "\n".join(errors[:5])
    return str(archive), detail


@_exclusive_backend("cutout")
def cutout_image(image_path: str):
    from gpu_cutout import model_path, prepare, process
    model = model_path()
    if not model.exists():
        raise gr.Error(f"未找到 ToonOut 抠图模型：{model}")
    target = OUTPUT_DIR / f"{Path(image_path).stem}_cutout.png"
    try:
        # Load (and warm up) first: with cudnn.benchmark enabled the very first
        # forward pass spends ~20-35 s searching for kernels, and that cost
        # belongs to model loading rather than to the reported frame time.
        prepare(model)
        elapsed = process(image_path, target, model)
    except Exception as exc:
        raise gr.Error(f"ToonOut · BiRefNet 动漫抠图失败：{exc}") from exc
    return str(target), f"ToonOut · BiRefNet · CUDA GPU · 推理 {elapsed:.2f} 秒，已输出透明 PNG。"


@_exclusive_backend("relight")
def relight_image(image_path: str):
    """Relight with the bundled Marigold surface-normal model."""
    try:
        import numpy as np
        from PIL import Image
    except ImportError as exc:
        raise gr.Error("重新打光需要安装 pillow、numpy。请重新运行 start.ps1。") from exc
    image = Image.open(image_path).convert("RGB")
    try:
        model = MODEL_DIR / "only-shadow-marigold-normals-v1-1"
        if not model.exists():
            raise RuntimeError(f"未找到 Marigold Normals 模型：{model}")
        from diffusers import MarigoldNormalsPipeline
        pipe = _marigold_pipeline()
        # Marigold is most reliable at its native working resolution. The
        # result is resized back to the original image for a lossless export.
        prediction = pipe(
            image,
            num_inference_steps=8,
            ensemble_size=1,
            processing_resolution=640,
        ).prediction[0]
        normals = np.asarray(prediction, dtype=np.float32)
        normals = normals / np.maximum(np.linalg.norm(normals, axis=2, keepdims=True), 1e-6)
        # PIL does not resize float RGB arrays directly; encode the signed
        # normal vector into 8-bit RGB for the resize, then decode it again.
        normal_img = Image.fromarray(np.uint8(np.clip((normals + 1.0) * 127.5, 0, 255)), mode="RGB")
        normals = np.asarray(normal_img.resize(image.size, Image.Resampling.BICUBIC), dtype=np.float32) / 127.5 - 1.0
        light_direction = np.asarray([0.35, 0.55, 0.76], dtype=np.float32)
        light_direction /= np.linalg.norm(light_direction)
        shade = np.clip(np.sum(normals * light_direction[None, None, :], axis=2), 0.0, 1.0)
        illumination = 0.58 + 0.72 * shade
        rgb = np.asarray(image, dtype=np.float32) * illumination[..., None]
        result = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), mode="RGB")
        note = "已使用 Marigold Normals 高质量表面法线完成重打光。"
    except Exception as exc:
        raise gr.Error(f"Marigold Normals 重打光失败：{exc}") from exc
    target = OUTPUT_DIR / f"{Path(image_path).stem}_relit.png"
    result.save(target)
    return str(target), note


@lru_cache(maxsize=1)
def _marigold_pipeline():
    """Load the local Marigold normals pipeline once per process."""
    import torch
    from diffusers import MarigoldNormalsPipeline
    model = MODEL_DIR / "only-shadow-marigold-normals-v1-1"
    use_cuda = torch.cuda.is_available()
    pipe = MarigoldNormalsPipeline.from_pretrained(
        str(model),
        variant="fp16" if use_cuda else None,
        torch_dtype=torch.float16 if use_cuda else torch.float32,
        local_files_only=True,
    )
    return pipe.to("cuda" if use_cuda else "cpu")


@_exclusive_backend("cutout")
def video_cutout(video_path: str | None, output_format: str = "透明 PNG 序列（ZIP · 无损）",
                 batch_size: int = CUTOUT_BATCH_SIZE, progress=gr.Progress()):
    from gpu_cutout import flush, model_path, prepare, process_prepared_batch
    import numpy as np
    import time
    from PIL import Image

    source = ensure_video(video_path)
    model = model_path()
    if not model.exists():
        raise gr.Error(f"未找到 ToonOut 抠图模型：{model}")
    shape = video_shape(source)
    total_hint = estimated_frame_count(source)
    if not shape or not total_hint:
        raise gr.Error("无法读取视频分辨率或总帧数。")
    try:
        # BiRefNet uses a 1024px input, so large batches can exhaust 8 GB VRAM.
        batch_size = max(1, min(CUTOUT_BATCH_MAX, int(batch_size)))
    except (TypeError, ValueError):
        batch_size = CUTOUT_BATCH_SIZE

    width, height = shape
    direct_video = output_format.startswith(("MOV", "WebM"))
    target = None
    encoder = None
    job = None
    out_dir = None
    if direct_video:
        if output_format.startswith("MOV"):
            target = OUTPUT_DIR / f"{Path(source).stem}_cutout_prores4444.mov"
            command = [FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                       "-s", f"{width}x{height}", "-framerate", f"{fps(source):.8f}", "-i", "-",
                       "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
                       "-vendor", "apl0", "-bits_per_mb", "8000", str(target)]
            detail = "MOV · ProRes 4444 + Alpha"
        else:
            target = OUTPUT_DIR / f"{Path(source).stem}_cutout_vp9.webm"
            command = [FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "rgba",
                       "-s", f"{width}x{height}", "-framerate", f"{fps(source):.8f}", "-i", "-",
                       "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0",
                       "-crf", "24", "-b:v", "0", "-row-mt", "1", str(target)]
            detail = "WebM · VP9 + Alpha（下游解码需支持 VP9 alpha）"
        if Path(target).exists():
            Path(target).unlink()
        encoder = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                   stderr=subprocess.PIPE)
    else:
        target = OUTPUT_DIR / f"{Path(source).stem}_cutout_frames.zip"
        job = Path(tempfile.mkdtemp(prefix="cutout_", dir=WORK_DIR))
        out_dir = job / "transparent"
        out_dir.mkdir()
        detail = "透明 PNG 序列 ZIP"

    progress(0, desc=f"加载 ToonOut · BiRefNet · CUDA GPU · batch {batch_size}")
    try:
        # Warm up the exact batch shape selected by the user so the first real
        # batch does not pay a second cuDNN autotuning cost.
        prepare(model, batch_size)
    except Exception as exc:
        if encoder is not None:
            encoder.terminate()
        if job is not None:
            shutil.rmtree(job, ignore_errors=True)
        raise gr.Error(f"CUDA 初始化失败：{exc}") from exc

    rgba_writer = _CutoutRGBAWriter(encoder, width, height) if direct_video else None
    started = time.perf_counter()
    progress(0.02, desc=f"ToonOut · 预取处理 · 0/{total_hint} 帧 · batch {batch_size}")
    processed = 0
    processing_failed = False
    stop_event = threading.Event()
    batches, producer, stop_event, sentinel = _cutout_batch_producer(
        source, width, height, total_hint, batch_size,
        output_dir=out_dir, stop_event=stop_event
    )
    try:
        while True:
            item = batches.get()
            if item is sentinel:
                break
            if isinstance(item, tuple) and item and item[0] is sentinel:
                raise item[1]
            first_index, last_index, frames, prepared, targets = item
            try:
                result = process_prepared_batch(
                    prepared, frames, width, height, targets, model,
                    return_alpha=direct_video
                )
            except Exception as exc:
                raise gr.Error(f"GPU 抠图在第 {first_index}–{last_index} 帧失败：{exc}") from exc

            if direct_video:
                _, alpha_bytes = result
                rgba_writer.put(frames, alpha_bytes)

            processed = last_index
            elapsed = time.perf_counter() - started
            rate = processed / max(elapsed, .001)
            remaining = (total_hint - processed) / rate if rate > 0 else 0
            progress(0.9 * min(processed / total_hint, 1),
                     desc=f"ToonOut · CUDA GPU · {processed}/{total_hint} 帧 · "
                          f"{rate:.1f} 帧/秒 · batch {batch_size} · 剩余 {remaining:.0f} 秒")
    except Exception:
        processing_failed = True
        raise
    finally:
        stop_event.set()
        producer.join(timeout=5)
        if processing_failed and encoder is not None and encoder.poll() is None:
            if rgba_writer is not None:
                rgba_writer.abort()
            encoder.terminate()
            try:
                encoder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                encoder.kill()
        if processing_failed and job is not None:
            shutil.rmtree(job, ignore_errors=True)

    if not direct_video:
        try:
            flush()
        except Exception as exc:
            raise gr.Error(f"抠图结果写盘失败：{exc}") from exc
    else:
        try:
            rgba_writer.close()
            encoder.stdin.close()
            encoder.stdin = None
            stderr = encoder.stderr.read().decode("utf-8", "replace") if encoder.stderr else ""
            if encoder.wait() != 0:
                raise gr.Error("Alpha 视频编码失败：" + stderr[-700:])
        finally:
            if encoder is not None and encoder.poll() is None:
                encoder.terminate()

    mask_seconds = time.perf_counter() - started
    if not direct_video:
        made = sorted(out_dir.glob("*.png"))
        if not made:
            shutil.rmtree(job, ignore_errors=True)
            raise gr.Error("抠图没有产生任何帧。")
        progress(0.92, desc=f"正在打包 PNG 序列 · 0/{len(made)} 帧")
        try:
            with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as zf:
                for index, file in enumerate(made, 1):
                    zf.write(file, file.name)
                    if index == 1 or index % 10 == 0 or index == len(made):
                        progress(0.92 + 0.07 * index / len(made),
                                 desc=f"正在打包 PNG 序列 · {index}/{len(made)} 帧")
        finally:
            shutil.rmtree(job, ignore_errors=True)

    if not Path(target).exists():
        raise gr.Error("抠图输出文件未生成。")
    size_mb = Path(target).stat().st_size / 1_000_000
    progress(1, desc="完成 · ToonOut · BiRefNet · CUDA GPU")
    return str(target), (
        f"ToonOut · BiRefNet · CUDA GPU · 已处理 {processed} 帧，抠图用时 {mask_seconds:.1f} 秒，"
        f"batch {batch_size}，输出 {detail}（{size_mb:.1f} MB）。"
    )


@_exclusive_backend("upscale")
def upscale_image(image_path: str, mode: str = "Real-CUGAN Pro 2×（动漫推荐）",
                  denoise: str = DEFAULT_DENOISE, tta: bool = False):
    model_set, scale = _resolve_preset(mode)
    if not Path(REALCUGAN).exists():
        raise gr.Error("Real-CUGAN Vulkan 运行器缺失，未切换到低质量放大。")
    try:
        return realcugan_image(image_path, scale, model_set, DENOISE_CHOICES.get(denoise, -1), bool(tta))
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"{_realcugan_label(model_set, scale)} 超分失败：{exc}") from exc


def realcugan_image(image_path: str, scale: int = 2, model_set: str = "pro",
                    noise: int = -1, tta: bool = False):
    directory = _realcugan_weights(model_set, scale, noise)
    label = _realcugan_label(model_set, scale)
    target = OUTPUT_DIR / f"{Path(image_path).stem}_realcugan{scale}x.png"
    command = [REALCUGAN, "-i", str(image_path), "-o", str(target),
               "-s", str(scale), "-n", str(noise), "-m", str(directory)]
    if tta:
        command.append("-x")
    result = run(command, capture_output=True, text=True, timeout=3600)
    if result.returncode != 0 or not target.exists():
        raise RuntimeError(result.stderr[-500:] or "Real-CUGAN 执行失败")
    note = f"✓ {label} 超分完成（降噪 {noise}{'，TTA 极致质量' if tta else ''}）"
    return str(target), note


def realcugan_video(source: str, scale: int = 2, model_set: str = "pro",
                    noise: int = -1, tta: bool = False, progress=gr.Progress()):
    """Upscale a whole video with Real-CUGAN.

    The frames directory is handed to the Vulkan runner in one invocation
    instead of one process per frame.  That loads the network once and lets the
    runner's own load/proc/save pipeline overlap, which measured ~2.5x faster
    than spawning a process per frame.
    """
    if not Path(REALCUGAN).exists():
        raise gr.Error("Real-CUGAN Vulkan 运行器未找到。")
    directory = _realcugan_weights(model_set, scale, noise)
    label = _realcugan_label(model_set, scale)

    job = Path(tempfile.mkdtemp(prefix="cugan_", dir=WORK_DIR))
    frames = job / "frames"; frames.mkdir()
    out = job / "out"; out.mkdir()

    total_hint = estimated_frame_count(source)
    progress(0.02, desc=f"{label} 解码准备 · {total_hint or '?'} 帧")
    extracted = run_ffmpeg_with_frame_progress(
        [FFMPEG, "-y", "-i", source, str(frames / "frame_%06d.png")],
        total_hint, progress, 0.02, 0.12, f"{label} 解码"
    )
    if extracted.returncode != 0:
        raise gr.Error("视频抽帧失败：" + extracted.stderr[-600:])

    total = len(list(frames.glob("*.png")))
    if not total:
        raise gr.Error("视频中没有可处理的帧。")

    progress(0.14, desc=f"{label} 超分 · 0/{total} 帧")
    command = [REALCUGAN, "-i", str(frames), "-o", str(out),
               "-s", str(scale), "-n", str(noise), "-m", str(directory)]
    if tta:
        command.append("-x")
    runner = subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                              text=True, encoding="utf-8", errors="replace")
    started = time.perf_counter()
    reported = -1
    while runner.poll() is None:
        made_now = len(list(out.glob("*.png")))
        if made_now != reported:
            reported = made_now
            elapsed = max(time.perf_counter() - started, 1e-6)
            rate = made_now / elapsed
            remaining = (total - made_now) / rate if rate > 0 else 0
            progress(0.14 + 0.62 * min(made_now / total, 1.0),
                     desc=f"{label} 超分 · {made_now}/{total} 帧 · "
                          f"{rate:.2f} 帧/秒 · 剩余 {remaining / 60:.1f} 分钟")
        time.sleep(0.5)
    runner_stderr = runner.stderr.read() if runner.stderr is not None else ""
    result = subprocess.CompletedProcess(command, runner.returncode, "", runner_stderr)
    if result.returncode != 0:
        raise gr.Error(f"{label} 视频超分失败：" + result.stderr[-600:])
    made = sorted(out.glob("*.png"))
    if not made:
        raise gr.Error(f"{label} 没有输出任何帧。")

    progress(0.78, desc=f"{label} 超分完成 · {len(made)}/{total} 帧，准备编码")
    target = OUTPUT_DIR / f"{Path(source).stem}_realcugan{scale}x.mp4"
    # An Nx upscale multiplies pixel count by N^2, so scale the source bitrate
    # the same way and keep a floor for clean anime edges.
    target_bitrate = max(8_000_000, int((video_bitrate(source) or 8_000_000) * scale * scale * 1.15))
    encoded = run_ffmpeg_with_frame_progress(
        [FFMPEG, "-y", "-framerate", str(fps(source)), "-i", str(out / "frame_%06d.png"),
         "-i", source, "-map", "0:v:0", "-map", "1:a?",
         "-c:v", "libx264", "-crf", "12", "-preset", "slow", "-tune", "animation",
         "-b:v", str(target_bitrate), "-maxrate", str(target_bitrate), "-bufsize", str(target_bitrate * 2),
         "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", "-movflags", "+faststart", str(target)],
        len(made), progress, 0.80, 0.98, f"{label} 编码"
    )
    if encoded.returncode != 0:
        raise gr.Error("视频编码失败：" + encoded.stderr[-600:])

    shutil.rmtree(job, ignore_errors=True)
    progress(1, desc=f"完成 · {label}")
    return str(target), (
        f"✓ {label} 视频超分完成（降噪 {noise}{'，TTA 极致质量' if tta else ''}），"
        f"已处理 {len(made)} 帧并保留原音轨。"
    )


@lru_cache(maxsize=1)
def _depth_large_pipeline():
    try:
        import torch
        from transformers import AutoModelForDepthEstimation
    except ImportError as exc:
        raise RuntimeError("Depth Anything V2 Large 需要 transformers 和 torch") from exc
    if not (DEPTH_LARGE_MODEL / "model.safetensors").exists():
        raise RuntimeError(f"未找到 Depth Anything V2 Large 模型：{DEPTH_LARGE_MODEL}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    model = AutoModelForDepthEstimation.from_pretrained(
        str(DEPTH_LARGE_MODEL), local_files_only=True, torch_dtype=dtype
    ).to(device).eval()
    return model, device


def _depth_large_batch(images):
    """Run several equally sized frames through Depth Anything in one call.

    The model still receives its native 518x518 input, but batching removes
    the per-frame CUDA launch and synchronisation overhead.  The output is
    converted back to each source frame's size before it leaves the GPU.
    """
    import numpy as np
    import torch
    from PIL import Image

    if not images:
        return []
    model, device = _depth_large_pipeline()
    mean = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    arrays = []
    for image in images:
        resized = image.resize((518, 518), Image.Resampling.BICUBIC)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        arrays.append(np.transpose((array - mean) / std, (2, 0, 1)))
    inputs = {"pixel_values": torch.from_numpy(np.ascontiguousarray(np.stack(arrays))).to(device)}
    with torch.inference_mode():
        predicted = model(**inputs).predicted_depth
        predicted = torch.nn.functional.interpolate(
            predicted.unsqueeze(1),
            size=(images[0].height, images[0].width),
            mode="bicubic", align_corners=False,
        )[:, 0]
    # Depth Anything runs in FP16 on CUDA.  Transfer that unchanged tensor and
    # widen on the CPU; widening before the copy doubles the transfer volume
    # without changing the resulting FP32 values.
    values = predicted.cpu().float().numpy()
    return [values[index] for index in range(len(images))]


def _depth_large_array(image):
    """Single-image compatibility wrapper around the batched implementation."""
    return _depth_large_batch([image])[0]


def _save_depth_array(depth, size, target, low=None, high=None):
    import numpy as np
    from PIL import Image
    Image.fromarray(_depth_uint8_array(depth, size, low, high), mode="L").save(target)


def _depth_uint8_array(depth, size, low=None, high=None):
    import numpy as np
    if low is None or high is None:
        low, high = np.percentile(depth, [1, 99])
    normalized = np.clip((depth - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
    # _depth_large_batch already upsamples to the source frame dimensions.
    # Avoid a second bicubic pass when the requested output size is identical;
    # this preserves the exact pixels and removes one CPU resize per frame.
    if normalized.shape[1] == size[0] and normalized.shape[0] == size[1]:
        return normalized
    from PIL import Image
    return np.asarray(Image.fromarray(normalized, mode="L").resize(size, Image.Resampling.BICUBIC))


def _depth_global_range(files, sample_limit: int = 12):
    """Estimate one depth range for a whole clip.

    Normalising every frame against its own 1st/99th percentile makes the
    output "pump" - a shot with a narrow depth spread gets stretched as hard as
    a wide one, so the grey level of an unchanged object drifts between frames.
    Pooling samples from across the clip and reusing that range keeps the
    mapping stable, which is what makes a depth *video* watchable.
    """
    import numpy as np
    from PIL import Image

    if not files:
        return None, None
    step = max(1, len(files) // sample_limit)
    sampled = files[::step][:sample_limit]
    pooled = []
    for file in sampled:
        with Image.open(file) as handle:
            depth = _depth_large_array(handle.convert("RGB"))
        # Subsample: enough for a stable percentile, cheap on memory.
        pooled.append(depth.ravel()[::53])
    values = np.concatenate(pooled)
    low, high = np.percentile(values, [1, 99])
    return float(low), float(high)


@_exclusive_backend("depth")
def depth_image(image_path: str):
    if not (DEPTH_LARGE_MODEL / "model.safetensors").exists():
        raise gr.Error(f"未找到 Depth Anything V2 Large 模型：{DEPTH_LARGE_MODEL}")
    try:
        from PIL import Image
    except ImportError as exc:
        raise gr.Error("深度图需要安装 Pillow。请重新运行 start.ps1。") from exc
    image = Image.open(image_path).convert("RGB")
    target = OUTPUT_DIR / f"{Path(image_path).stem}_depth.png"
    try:
        _save_depth_array(_depth_large_array(image), image.size, target)
    except Exception as exc:
        raise gr.Error(f"Depth Anything V2 Large 深度图失败：{exc}") from exc
    return str(target), "Depth Anything V2 Large 高质量深度图已生成。"


# Video depth always maps the whole clip through one fixed grey range.  Testing
# per-frame normalisation showed it buys slightly stronger single-frame contrast
# but re-maps identical depths to different greys between shots, which makes the
# result unusable for compositing and causes the output to "pump".
CUTOUT_OUTPUTS = [
    "透明 PNG 序列（ZIP · 无损）",
    "MOV（ProRes 4444 + Alpha · 剪辑用）",
    "WebM（VP9 + Alpha · 体积小）",
]


@_exclusive_backend("depth")
def depth_video(video_path: str | None, batch_size: int = DEPTH_BATCH_SIZE,
                progress=gr.Progress()):
    source = ensure_video(video_path)
    if not (DEPTH_LARGE_MODEL / "model.safetensors").exists():
        raise gr.Error(f"未找到 Depth Anything V2 Large 模型：{DEPTH_LARGE_MODEL}")
    try:
        from PIL import Image
    except ImportError as exc:
        raise gr.Error("深度图需要安装 Pillow。请重新运行 start.ps1。") from exc
    shape = video_shape(source)
    if not shape:
        raise gr.Error("无法读取视频分辨率。")
    width, height = shape
    total_hint = estimated_frame_count(source)
    if not total_hint:
        raise gr.Error("无法读取视频总帧数。")
    try:
        # Keep the UI/API value inside the tested range for an 8 GB GPU.
        batch_size = max(1, min(DEPTH_BATCH_MAX, int(batch_size)))
    except (TypeError, ValueError):
        batch_size = DEPTH_BATCH_SIZE
    encoder = None
    frame_writer = None
    try:
        sample_limit = 12
        sample_step = max(1, total_hint // sample_limit)
        sample_images = []
        progress(0.02, desc=f"Depth Anything · 管道解码采样 · 0/{sample_limit} 帧")
        for frame in raw_video_sample_frames(source, width, height, sample_step):
            if len(sample_images) >= sample_limit:
                break
            sample_images.append(Image.fromarray(frame, mode="RGB"))
            progress(0.02 + 0.12 * len(sample_images) / sample_limit,
                     desc=f"Depth Anything · 管道采样 {len(sample_images)}/{sample_limit} 帧")
        sampled = []
        for start in range(0, len(sample_images), batch_size):
            sampled.extend(_depth_large_batch(sample_images[start:start + batch_size]))
        # The PIL samples are no longer needed once their depth arrays exist;
        # release their source-frame memory before the full video pass.
        del sample_images
        if not sampled:
            raise gr.Error("视频中没有可处理的帧。")
        import numpy as np
        values = np.concatenate([depth.ravel()[::53] for depth in sampled])
        low, high = np.percentile(values, [1, 99])

        target = OUTPUT_DIR / f"{Path(source).stem}_depth.mp4"
        target_rate = fps(source)
        encoder = subprocess.Popen(
            [FFMPEG, "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
             "-s", f"{width}x{height}", "-framerate", f"{target_rate:.8f}", "-i", "-",
             "-i", source, "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-crf", "10",
             "-preset", "slow", "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", str(target)],
            stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        def encode_depth(depth):
            return np.ascontiguousarray(
                _depth_uint8_array(depth, (width, height), low, high)
            ).tobytes()

        frame_writer = _FramePipeWriter(encoder, convert=encode_depth,
                                        name="depth-frame-writer")
        progress(0.16, desc=f"Depth Anything V2 Large · 推理 0/{total_hint} 帧 · batch {batch_size}")
        started = time.perf_counter()
        depth_batches, producer, producer_stop, producer_sentinel, producer_errors = _depth_batch_producer(
            source, width, height, batch_size
        )
        processed = 0
        while True:
            frame_batch = depth_batches.get()
            if frame_batch is producer_sentinel:
                break
            depths = _depth_large_batch(frame_batch)
            for depth in depths:
                frame_writer.put(depth)
            processed += len(frame_batch)
            elapsed = max(time.perf_counter() - started, 1e-6)
            rate = processed / elapsed
            remaining = (total_hint - processed) / rate if rate > 0 else 0
            progress(min(0.88, 0.16 + 0.72 * processed / total_hint),
                     desc=f"Depth Anything V2 Large · {processed}/{total_hint} 帧 · "
                          f"{rate:.2f} 帧/秒 · batch {batch_size} · 剩余 {max(0, remaining) / 60:.1f} 分钟")
        producer.join(timeout=5)
        if producer_errors:
            raise producer_errors[0]
        frame_writer.close()
        frame_writer = None
        encoder.stdin.close()
        encoder.stdin = None
        stderr = encoder.stderr.read().decode("utf-8", "replace") if encoder.stderr else ""
        if encoder.wait() != 0:
            raise gr.Error("深度图视频编码失败：" + stderr[-700:])
        return str(target), (
            f"已处理 {processed} 帧，使用 Depth Anything V2 Large 输出灰度深度图视频"
            f"（batch {batch_size}；全片统一深度范围 {low:.2f}–{high:.2f}，跨镜头灰阶一致）。"
        )
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Depth Anything V2 Large 视频深度图失败：{exc}") from exc
    finally:
        if 'frame_writer' in locals() and frame_writer is not None:
            frame_writer.abort()
        if 'producer_stop' in locals() and producer_stop:
            producer_stop.set()
        if 'producer' in locals() and producer:
            producer.join(timeout=5)
        if 'encoder' in locals() and encoder is not None:
            if encoder.stdin:
                encoder.stdin.close()
            if encoder.poll() is None:
                encoder.terminate()
            try:
                encoder.wait(timeout=5)
            except subprocess.TimeoutExpired:
                encoder.kill()
                encoder.wait()



from types import SimpleNamespace
from ui import build_ui, CSS, THEME
from lifecycle import request_exit, watch_launcher

demo = build_ui(SimpleNamespace(**globals()))

if __name__ == "__main__":
    watch_launcher()
    demo.launch(inbrowser=False, server_name="127.0.0.1", server_port=7860, show_error=True,
                theme=THEME, css=CSS, footer_links=[])
