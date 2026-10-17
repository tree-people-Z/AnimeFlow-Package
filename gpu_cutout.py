"""CUDA matting for the bundled ToonOut model, running on PyTorch.

The model is a BiRefNet fine-tuned for anime line art ("ToonOut", Muratori &
Seytre 2025).  Inference runs through PyTorch so it shares the CUDA 12.8
runtime that the rest of AnimeFlow already uses and works on Blackwell
(sm_120) GPUs.

An earlier build ran the same weights through onnxruntime-gpu.  That path is
no longer used: the packaged onnxruntime CUDA provider creates a
CUDAExecutionProvider session on Blackwell but then dies with an access
violation (0xC0000005) on the first inference for every session-option
combination, and the CPU fallback costs ~7.9 s per frame.  PyTorch does the
same work in ~0.22 s per frame on the same GPU.

Native GPU work still happens in a separate process so that a driver-level
crash cannot take the UI down with it.
"""
from __future__ import annotations

import atexit
import multiprocessing
import queue
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

# The packaged embeddable Python takes sys.path from python313._pth and does not
# add the running script's directory, so make the app root explicit; the spawned
# GPU worker re-imports this module in a fresh interpreter.
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

_CUTOUT_MODEL_DIR = APP_DIR / "models" / "toonout-pytorch"
TOONOUT_WEIGHTS = _CUTOUT_MODEL_DIR / "birefnet_finetuned_toonout.pth"

# The working resolution of BiRefNet's Swin-L backbone.
_INPUT_SIZE = 1024
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Checkpoints are wrapped by torch.compile ("_orig_mod.") and/or DataParallel
# ("module."), sometimes both at once, so strip repeatedly until stable.
_WRAPPER_PREFIXES = ("_orig_mod.", "module.", "model.")

_lock = threading.Lock()
_warmed_batches = set()


def model_path() -> Path:
    """Return the ToonOut weights that the application should use."""
    return TOONOUT_WEIGHTS


def _strip_wrappers(state: dict) -> dict:
    # Most exported checkpoints are already clean.  Returning the original
    # mapping avoids duplicating hundreds of megabytes during startup.
    if not any(
        isinstance(key, str) and key.startswith(_WRAPPER_PREFIXES)
        for key in state
    ):
        return state
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


@lru_cache(maxsize=1)
def session(weights_path: str):
    """Load the ToonOut network onto the GPU once per process."""
    import torch

    path = Path(weights_path)
    if not path.is_file():
        raise RuntimeError(f"未找到 ToonOut 抠图模型：{path}")
    if not torch.cuda.is_available():
        raise RuntimeError(
            "抠图需要 CUDA GPU；未检测到可用的 CUDA 设备，未切换到 CPU。"
            f" torch={torch.__version__}"
        )

    from vendor.toonout_birefnet import BiRefNet, BiRefNetConfig

    try:
        # mmap defers storage reads and reduces the peak CPU memory/IO burst.
        raw = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except (TypeError, RuntimeError):
        # Older packaged torch builds may not support mmap for this checkpoint.
        raw = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(raw, dict) and "state_dict" in raw:
        raw = raw["state_dict"]
    state = _strip_wrappers(raw)

    model = BiRefNet(config=BiRefNetConfig(bb_pretrained=False))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "ToonOut 权重与网络结构不匹配："
            f"缺失 {len(missing)} 项、多余 {len(unexpected)} 项。"
            + (f" 首个缺失项 {missing[0]}。" if missing else "")
        )

    # Convert before/while moving to CUDA so loading does not briefly keep a
    # second full-precision copy on the GPU.
    model.eval().to(device="cuda", dtype=torch.float16)
    torch.backends.cudnn.benchmark = True
    return model


def _prepare(weights_path: str, batch_size: str = "1"):
    with _lock:
        model = session(str(weights_path))
        selected_batch = max(1, int(batch_size))
        warmup_key = (str(weights_path), selected_batch)
        # A cached worker may serve several cutout jobs.  Do not repeat the
        # expensive cuDNN warm-up for a batch shape already tuned in this
        # process; a newly selected batch shape is warmed exactly once.
        if warmup_key not in _warmed_batches:
            _warmup(model, selected_batch)
            _warmed_batches.add(warmup_key)
        return "ToonOut · BiRefNet · CUDA"


def _warmup(model, batch_size: int = 1):
    """Run one throwaway pass so cuDNN autotuning is not billed to frame 1.

    With cudnn.benchmark enabled the first forward pass on a new input shape
    spends ~20 s searching for kernels.  Paying that during load keeps the
    reported per-frame time meaningful.
    """
    import torch

    try:
        batch_size = max(1, int(batch_size))
        with torch.inference_mode():
            model(torch.zeros(batch_size, 3, _INPUT_SIZE, _INPUT_SIZE,
                              device="cuda", dtype=torch.float16))
        torch.cuda.synchronize()
    except Exception:
        # A failed warm-up must not block the real request; the real frame
        # will surface any genuine problem with a useful message.
        pass


def _run_model(model, tensor):
    import torch

    with torch.inference_mode():
        logits = model(tensor)[-1]
        return logits.sigmoid()


def _save_rgb_batch(images, alpha, targets):
    """Write one completed mask batch on the CPU writer thread."""
    import numpy as np
    from PIL import Image

    for image, mask_array, target in zip(images, alpha, targets):
        mask = Image.fromarray(np.clip(mask_array * 255.0, 0, 255).astype(np.uint8), mode="L")
        if mask.size != image.size:
            mask = mask.resize(image.size, Image.Resampling.LANCZOS)
        image.putalpha(mask)
        image.save(target, compress_level=1)


# GPU inference and PNG encoding use different resources.  A small bounded
# queue lets the next batch enter the GPU while the previous batch is being
# saved, without allowing decoded frames to grow without limit in RAM.
_writer_queue = None
_writer_thread = None
_writer_error = None
_writer_lock = threading.Lock()


def _writer_loop():
    global _writer_error
    while True:
        item = _writer_queue.get()
        try:
            if item is None:
                return
            _save_rgb_batch(*item)
        except Exception as exc:
            with _writer_lock:
                if _writer_error is None:
                    _writer_error = exc
        finally:
            _writer_queue.task_done()


def _ensure_writer():
    global _writer_queue, _writer_thread, _writer_error
    if _writer_queue is None:
        _writer_queue = queue.Queue(maxsize=2)
        _writer_error = None
    if _writer_thread is None or not _writer_thread.is_alive():
        _writer_thread = threading.Thread(target=_writer_loop, name="toonout-png-writer", daemon=True)
        _writer_thread.start()


def _check_writer_error():
    with _writer_lock:
        error = _writer_error
    if error is not None:
        raise RuntimeError(f"抠图结果写盘失败：{error}") from error


def _enqueue_rgb_batch(images, alpha, targets):
    _check_writer_error()
    _ensure_writer()
    # The bounded queue is intentional: if disk encoding is slower than GPU,
    # backpressure keeps memory stable instead of accumulating whole-video data.
    _writer_queue.put((images, alpha, targets))


def _flush_writer():
    if _writer_queue is None:
        return
    _writer_queue.join()
    _check_writer_error()


def _prepare_pinned_tensor(prepared_frames):
    """Build a pinned CPU tensor so the H2D copy can be non-blocking."""
    import numpy as np
    import torch

    arrays = [
        np.frombuffer(frame, dtype=np.float32).reshape(3, _INPUT_SIZE, _INPUT_SIZE)
        for frame in prepared_frames
    ]
    source = torch.from_numpy(np.ascontiguousarray(np.stack(arrays)))
    try:
        pinned = torch.empty_like(source, pin_memory=True)
        pinned.copy_(source)
        return pinned
    except RuntimeError:
        # Pinned allocation is optional; keep a safe fallback for systems that
        # disable page-locked host memory.
        return source


def _process_prepared_batch_bytes(prepared_frames, frames, width: int, height: int,
                                  targets, weights_path: str, tta: str = "False",
                                  return_alpha: str = "False"):
    """Infer a batch whose resize/normalisation was prepared by the producer.

    Preparing the next batch in the app process overlaps CPU work with the
    worker's current GPU forward.  ``return_alpha`` is used by direct video
    encoding, which avoids creating temporary transparent PNG files.
    """
    import numpy as np
    import torch
    from PIL import Image

    width = int(width)
    height = int(height)
    if len(prepared_frames) != len(frames) or len(frames) != len(targets):
        raise RuntimeError("抠图预处理批次参数数量不一致。")
    expected_prepared = 3 * _INPUT_SIZE * _INPUT_SIZE * 4
    should_return_alpha = str(return_alpha).lower() in ("1", "true", "yes")
    images = []
    expected = width * height * 3
    for rgb_bytes in frames:
        raw = np.frombuffer(rgb_bytes, dtype=np.uint8)
        if raw.size != expected:
            raise RuntimeError(f"原始帧大小异常：{raw.size}，预期 {expected}")
        if not should_return_alpha:
            images.append(Image.fromarray(raw.reshape((height, width, 3)), mode="RGB"))
    for prepared in prepared_frames:
        if len(prepared) != expected_prepared:
            raise RuntimeError(f"预处理帧大小异常：{len(prepared)}，预期 {expected_prepared}")

    tensor = _prepare_pinned_tensor(prepared_frames)
    tensor = tensor.to("cuda", dtype=torch.float16, non_blocking=True)
    use_tta = str(tta).lower() in ("1", "true", "yes")
    with _lock:
        model = session(str(weights_path))
        started = time.perf_counter()
        alpha = _run_model(model, tensor)
        if use_tta:
            flipped = _run_model(model, torch.flip(tensor, dims=[3]))
            alpha = (alpha + torch.flip(flipped, dims=[3])) * 0.5
        if not torch.isfinite(alpha).all():
            raise RuntimeError("GPU 输出出现无效数值，已停止抠图。")
        # FP16 is enough for the later 8-bit alpha video conversion and halves
        # the IPC payload.  Keep the model's FP16 result on the GPU until the
        # copy; widening to FP32 first only increases transfer work.
        alpha_tensor = alpha[:, 0]
        if should_return_alpha:
            alpha = alpha_tensor.cpu().numpy().astype(np.float16, copy=False)
        else:
            alpha = alpha_tensor.float().cpu().numpy()
        elapsed = time.perf_counter() - started

    if should_return_alpha:
        return elapsed, alpha.tobytes()
    _enqueue_rgb_batch(images, alpha, targets)
    return elapsed


def _process_rgb_batch_bytes(frames, width: int, height: int, targets,
                             weights_path: str, tta: str = "False"):
    """Run a batch of RGB24 frames in one GPU forward pass.

    ``frames`` remain raw RGB bytes across the worker IPC boundary, so this
    improves launch efficiency without changing the source or output pixels.
    The batch is intentionally bounded by the caller because BiRefNet at
    1024px has a high activation-memory cost on 8 GB cards.
    """
    import numpy as np
    import torch
    from PIL import Image

    width = int(width)
    height = int(height)
    if len(frames) != len(targets) or not frames:
        raise RuntimeError("抠图批次参数为空或帧数与输出数不一致。")
    expected = width * height * 3
    images = []
    arrays = []
    mean = np.asarray(_IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(_IMAGENET_STD, dtype=np.float32)
    for rgb_bytes in frames:
        frame = np.frombuffer(rgb_bytes, dtype=np.uint8)
        if frame.size != expected:
            raise RuntimeError(f"原始帧大小异常：{frame.size}，预期 {expected}")
        rgb = Image.fromarray(frame.reshape((height, width, 3)), mode="RGB")
        images.append(rgb)
        prepared = np.asarray(
            rgb.resize((_INPUT_SIZE, _INPUT_SIZE), Image.Resampling.BICUBIC), dtype=np.float32
        ) / 255.0
        arrays.append(np.ascontiguousarray(((prepared - mean) / std).transpose(2, 0, 1)))
    tensor = torch.from_numpy(np.ascontiguousarray(np.stack(arrays))).to(
        "cuda", dtype=torch.float16
    )

    use_tta = str(tta).lower() in ("1", "true", "yes")
    with _lock:
        model = session(str(weights_path))
        started = time.perf_counter()
        alpha = _run_model(model, tensor)
        if use_tta:
            flipped = _run_model(model, torch.flip(tensor, dims=[3]))
            alpha = (alpha + torch.flip(flipped, dims=[3])) * 0.5
        if not torch.isfinite(alpha).all():
            raise RuntimeError("GPU 输出出现无效数值，已停止抠图。")
        alpha = alpha[:, 0].float().cpu().numpy()
        elapsed = time.perf_counter() - started

    _enqueue_rgb_batch(images, alpha, targets)
    return elapsed


def _process(image_path: str, target: str, weights_path: str, tta: str = "False"):
    """Matte one frame and write it as a transparent PNG."""
    import numpy as np
    import torch
    from PIL import Image

    with Image.open(image_path) as original:
        rgb = original.convert("RGB")
    original_size = rgb.size

    prepared = np.asarray(
        rgb.resize((_INPUT_SIZE, _INPUT_SIZE), Image.Resampling.BICUBIC), dtype=np.float32
    ) / 255.0
    mean = np.asarray(_IMAGENET_MEAN, dtype=np.float32)
    std = np.asarray(_IMAGENET_STD, dtype=np.float32)
    normalized = (prepared - mean) / std
    tensor = torch.from_numpy(
        np.ascontiguousarray(normalized.transpose(2, 0, 1))[None]
    ).to("cuda", dtype=torch.float16)

    use_tta = str(tta).lower() in ("1", "true", "yes")

    with _lock:
        model = session(str(weights_path))
        started = time.perf_counter()
        alpha = _run_model(model, tensor)
        if use_tta:
            # Averaging the mask with a horizontally flipped pass tightens the
            # silhouette on thin anime line art.  Costs one extra forward pass.
            flipped = _run_model(model, torch.flip(tensor, dims=[3]))
            alpha = (alpha + torch.flip(flipped, dims=[3])) * 0.5
        if not torch.isfinite(alpha).all():
            raise RuntimeError("GPU 输出出现无效数值，已停止抠图。")
        alpha = alpha[0, 0].float().cpu().numpy()
        elapsed = time.perf_counter() - started

    mask = Image.fromarray(np.clip(alpha * 255.0, 0, 255).astype(np.uint8), mode="L")
    if mask.size != original_size:
        mask = mask.resize(original_size, Image.Resampling.LANCZOS)
    rgb.putalpha(mask)
    rgb.save(target, compress_level=1)
    return elapsed


def _process_rgb_bytes(rgb_bytes: bytes, width: int, height: int, target: str,
                       weights_path: str, tta: str = "False"):
    """Process one RGB24 frame received over IPC, avoiding input PNG I/O."""
    return _process_rgb_batch_bytes([rgb_bytes], width, height, [target], weights_path, tta)


def _release_session():
    """Unload the cached CUDA model without stopping the worker process."""
    import gc
    import torch

    _flush_writer()
    _warmed_batches.clear()
    session.cache_clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# Keep native GPU runtime failures out of the UI process.
_worker_process = None
_connection = None
_rpc_lock = threading.Lock()


def _worker(connection, parent_id):
    import os

    from lifecycle import watch_launcher

    os.environ["ANIMEFLOW_PARENT_PID"] = str(parent_id)
    watch_launcher()
    while True:
        try:
            command, args = connection.recv()
            if command == "prepare":
                _prepare(*args)
                value = "ToonOut · CUDA"
            elif command == "flush":
                _flush_writer()
                value = "ToonOut · 输出队列已完成"
            elif command == "release":
                _release_session()
                value = "ToonOut · CUDA · 已释放"
            elif command == "process_rgb":
                value = _process_rgb_bytes(*args)
            elif command == "process_rgb_batch":
                value = _process_rgb_batch_bytes(*args)
            elif command == "process_prepared_batch":
                value = _process_prepared_batch_bytes(*args)
            else:
                value = _process(*args)
            connection.send((True, value))
        except EOFError:
            return
        except Exception as exc:
            connection.send((False, f"{type(exc).__name__}: {exc}"))


def _shutdown_worker():
    if _worker_process is not None and _worker_process.is_alive():
        _worker_process.terminate()
        _worker_process.join(timeout=5)


atexit.register(_shutdown_worker)


def _request(command, *args):
    global _worker_process, _connection
    import os

    with _rpc_lock:
        if _worker_process is None or not _worker_process.is_alive():
            context = multiprocessing.get_context("spawn")
            _connection, child = context.Pipe()
            _worker_process = context.Process(target=_worker, args=(child, os.getpid()), daemon=True)
            _worker_process.start()
            child.close()
        try:
            payload = args if command in ("process_rgb", "process_rgb_batch", "process_prepared_batch") else tuple(str(v) for v in args)
            _connection.send((command, payload))
            while not _connection.poll(0.2):
                if not _worker_process.is_alive():
                    raise RuntimeError(
                        f"GPU 推理进程异常退出（{_worker_process.exitcode}）。未切换 CPU；"
                        "请检查显卡驱动与 CUDA 运行库。"
                    )
            ok, value = _connection.recv()
        except (EOFError, BrokenPipeError, OSError) as exc:
            raise RuntimeError("GPU 推理进程已停止，未切换 CPU；应用界面仍可继续使用。") from exc
        if not ok:
            raise RuntimeError(value)
        return value


def prepare(weights_path, batch_size: int = 1) -> str:
    """Load and warm the model in the worker for the selected batch shape."""
    return _request("prepare", weights_path, int(batch_size))


def flush():
    """Wait until all asynchronously encoded PNG batches are written."""
    global _worker_process, _connection
    with _rpc_lock:
        if _worker_process is None or not _worker_process.is_alive():
            return
        try:
            _connection.send(("flush", tuple()))
            while not _connection.poll(0.2):
                if not _worker_process.is_alive():
                    return
            ok, value = _connection.recv()
            if not ok:
                raise RuntimeError(value)
        except (EOFError, BrokenPipeError, OSError):
            return


def release():
    """Release the worker's cached model if the worker is already running."""
    global _worker_process, _connection
    with _rpc_lock:
        if _worker_process is None or not _worker_process.is_alive():
            return
        try:
            _connection.send(("release", tuple()))
            while not _connection.poll(0.2):
                if not _worker_process.is_alive():
                    return
            ok, value = _connection.recv()
            if not ok:
                raise RuntimeError(value)
        except (EOFError, BrokenPipeError, OSError):
            # The worker has already gone away, so its CUDA context is gone too.
            return


def process(image_path, target, weights_path, tta: bool = False):
    return _request("process", image_path, target, weights_path, bool(tta))


def process_rgb(frame, width, height, target, weights_path, tta: bool = False):
    """Send one RGB24 frame directly to the GPU worker without a source file."""
    return _request("process_rgb", (frame.tobytes() if hasattr(frame, "tobytes") else frame),
                    int(width), int(height), str(target), str(weights_path), bool(tta))


def process_rgb_batch(frames, width, height, targets, weights_path, tta: bool = False):
    """Send several RGB24 frames to the worker for one batched inference."""
    payload = [frame.tobytes() if hasattr(frame, "tobytes") else frame for frame in frames]
    return _request("process_rgb_batch", payload, int(width), int(height),
                    [str(target) for target in targets], str(weights_path), bool(tta))


def process_prepared_batch(prepared_frames, frames, width, height, targets,
                           weights_path, tta: bool = False,
                           return_alpha: bool = False):
    """Infer producer-prepared frames, optionally returning masks for FFmpeg."""
    prepared = [frame.tobytes() if hasattr(frame, "tobytes") else frame
                for frame in prepared_frames]
    raw = [frame.tobytes() if hasattr(frame, "tobytes") else frame for frame in frames]
    return _request(
        "process_prepared_batch", prepared, raw, int(width), int(height),
        [str(target) for target in targets], str(weights_path), bool(tta),
        bool(return_alpha)
    )
