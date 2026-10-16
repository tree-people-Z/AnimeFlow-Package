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


def model_path() -> Path:
    """Return the ToonOut weights that the application should use."""
    return TOONOUT_WEIGHTS


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

    model.eval().to("cuda").half()
    torch.backends.cudnn.benchmark = True
    return model


def _prepare(weights_path: str):
    with _lock:
        model = session(str(weights_path))
        _warmup(model)
        return "ToonOut · BiRefNet · CUDA"


def _warmup(model):
    """Run one throwaway pass so cuDNN autotuning is not billed to frame 1.

    With cudnn.benchmark enabled the first forward pass on a new input shape
    spends ~20 s searching for kernels.  Paying that during load keeps the
    reported per-frame time meaningful.
    """
    import torch

    try:
        with torch.inference_mode():
            model(torch.zeros(1, 3, _INPUT_SIZE, _INPUT_SIZE, device="cuda", dtype=torch.float16))
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
            _connection.send((command, tuple(str(v) for v in args)))
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


def prepare(weights_path) -> str:
    """Load the model in the worker so the first frame does not pay for it."""
    return _request("prepare", weights_path)


def process(image_path, target, weights_path, tta: bool = False):
    return _request("process", image_path, target, weights_path, bool(tta))
