from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import zipfile
import re
import sys
from functools import lru_cache
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
    before = frame_count(source)
    job = Path(tempfile.mkdtemp(prefix="animeflow_", dir=WORK_DIR))
    extension = "mkv" if output_format == "无损中间文件（MKV）" else "mp4"
    target = OUTPUT_DIR / f"{Path(source).stem}_dedup.{extension}"
    # mpdecimate detects repeated/near-repeated animation cels.  Re-timing the
    # remaining frames makes them a clean input sequence for interpolation.
    hi = max(24, int(sensitivity * 64))
    lo = max(12, int(sensitivity * 32))
    vf = f"mpdecimate=hi={hi}:lo={lo}:frac=0.1,setpts=N/{source_fps}/TB"
    codec = ["-c:v", "ffv1"] if extension == "mkv" else ["-c:v", "libx264", "-crf", "12", "-preset", "medium"]
    command = [FFMPEG, "-y", "-i", source, "-an", "-vf", vf, "-fps_mode", "vfr", *codec, str(target)]
    progress(0.2, desc="正在识别并移除重复帧")
    result = run(command)
    if result.returncode != 0:
        raise gr.Error("FFmpeg 处理失败：" + result.stderr[-700:])
    after = frame_count(str(target))
    removed = (before - after) if before is not None and after is not None else None
    detail = f"原始 {before or '?'} 帧 → 保留 {after or '?'} 帧"
    if removed is not None:
        detail += f"，移除 {removed} 帧（{removed / max(before, 1):.1%}）"
    detail += f"\n源帧率 {source_fps:.2f} FPS。建议将结果交给补帧引擎生成 2× 或 4× 帧率。"
    progress(1, desc="完成")
    return str(target), detail


def extract_frames(video_path: str | None, every: int, fmt: str, progress=gr.Progress()):
    source = ensure_video(video_path)
    job = Path(tempfile.mkdtemp(prefix="frames_", dir=WORK_DIR))
    ext = "png" if fmt == "PNG（无损）" else "jpg"
    pattern = str(job / f"frame_%06d.{ext}")
    vf = f"select=not(mod(n\\,{max(1, every)}))"
    command = [FFMPEG, "-y", "-i", source, "-vf", vf, "-vsync", "vfr"]
    if ext == "jpg":
        command += ["-q:v", "2"]
    command += [pattern]
    progress(0.2, desc="正在抽取画面")
    result = run(command)
    if result.returncode != 0:
        raise gr.Error("FFmpeg 抽帧失败：" + result.stderr[-700:])
    archive = OUTPUT_DIR / f"{Path(source).stem}_frames.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in sorted(job.glob(f"*.{ext}")):
            zf.write(file, file.name)
    files = sorted(job.glob(f"*.{ext}"))
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
    progress(1, desc="完成")
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
    from PIL import Image

    label = "GMFSS Fortuna + RIFE 4.26-heavy"
    suffix = "gmfss-rife426"

    job = Path(tempfile.mkdtemp(prefix="vfiplus_", dir=WORK_DIR))
    frames_dir, output_dir = job / "input", job / "output"
    frames_dir.mkdir(); output_dir.mkdir()
    target = OUTPUT_DIR / f"{Path(source).stem}_{factor}x_{suffix}.mp4"
    try:
        extracted = run([FFMPEG, "-y", "-i", source, "-map", "0:v:0", "-vsync", "0",
                         str(frames_dir / "frame_%08d.png")])
        if extracted.returncode != 0:
            raise gr.Error("解码视频失败：" + extracted.stderr[-700:])
        frames = sorted(frames_dir.glob("*.png"))
        if len(frames) < 2:
            raise gr.Error("视频至少需要两帧才能补帧。")

        progress(0.03, desc=f"加载 {label} · CUDA")

        def source_frames():
            for path in frames:
                with Image.open(path) as handle:
                    yield np.asarray(handle.convert("RGB"))

        counter = {"n": 0}

        def emit(frame):
            counter["n"] += 1
            Image.fromarray(frame).save(output_dir / f"frame_{counter['n']:08d}.png")

        vfi_plus.interpolate_stream(source_frames(), int(factor), emit,
                                    progress=progress, total_hint=len(frames))

        if counter["n"] < 2:
            raise gr.Error("没有生成足够的画面。")

        progress(0.95, desc="正在编码")
        target_rate = fps(source) * factor
        target_bitrate = interpolation_bitrate(source, factor)
        encoded = run([FFMPEG, "-y", "-framerate", f"{target_rate:.8f}", "-start_number", "1",
                       "-i", str(output_dir / "frame_%08d.png"), "-i", source,
                       "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-crf", "10",
                       "-preset", "slow", "-tune", "animation",
                       "-b:v", str(target_bitrate), "-maxrate", str(target_bitrate),
                       "-bufsize", str(target_bitrate * 2),
                       "-pix_fmt", "yuv420p", "-c:a", "copy", "-movflags", "+faststart", str(target)])
        if encoded.returncode != 0:
            raise gr.Error("补帧结果编码失败：" + encoded.stderr[-700:])
        progress(1, desc=f"完成 · {label}")
        return str(target), (
            f"实际模型：{label} · CUDA。"
            f"{len(frames)} → {counter['n']} 帧，{fps(source):.2f} → {target_rate:.2f} FPS，"
            f"目标视频码率 {target_bitrate/1_000_000:.1f} Mbps。"
        )
    finally:
        shutil.rmtree(job, ignore_errors=True)


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


def video_cutout(video_path: str | None, output_format: str = "透明 PNG 序列（ZIP · 无损）",
                 progress=gr.Progress()):
    from gpu_cutout import model_path, prepare, process
    import time
    source = ensure_video(video_path)
    model = model_path()
    if not model.exists():
        raise gr.Error(f"未找到 ToonOut 抠图模型：{model}")
    progress(0, desc="加载 ToonOut · BiRefNet · CUDA GPU")
    try:
        prepare(model)
    except Exception as exc:
        raise gr.Error(f"CUDA 初始化失败：{exc}") from exc
    job = Path(tempfile.mkdtemp(prefix="cutout_", dir=WORK_DIR))
    frames = job / "frames"; frames.mkdir()
    progress(0, desc="正在解码视频帧")
    result = run([FFMPEG, "-y", "-i", source, str(frames / "frame_%06d.png")])
    if result.returncode != 0:
        raise gr.Error("视频抽帧失败：" + result.stderr[-500:])
    files = sorted(frames.glob("*.png"))
    if not files:
        raise gr.Error("视频中没有可处理的帧。")
    out_dir = job / "transparent"; out_dir.mkdir()
    started = time.perf_counter()
    for index, file in enumerate(files, 1):
        try:
            process(file, out_dir / file.name, model)
        except Exception as exc:
            raise gr.Error(f"GPU 抠图在第 {index} 帧失败：{exc}") from exc
        elapsed = time.perf_counter() - started
        rate = index / max(elapsed, .001)
        remaining = (len(files) - index) / rate
        progress(0.9 * index / len(files),
                 desc=f"ToonOut · CUDA GPU · {index}/{len(files)} 帧 · {rate:.1f} 帧/秒 · 剩余 {remaining:.0f} 秒")
    mask_seconds = time.perf_counter() - started

    made = sorted(out_dir.glob("*.png"))
    if not made:
        raise gr.Error("抠图没有产生任何帧。")
    sequence = str(out_dir / "frame_%06d.png")

    if output_format.startswith("MOV"):
        target = OUTPUT_DIR / f"{Path(source).stem}_cutout_prores4444.mov"
        progress(0.92, desc="正在编码 ProRes 4444 + Alpha")
        command = [FFMPEG, "-y", "-framerate", f"{fps(source):.8f}", "-i", sequence,
                   "-c:v", "prores_ks", "-profile:v", "4444", "-pix_fmt", "yuva444p10le",
                   "-vendor", "apl0", "-bits_per_mb", "8000", str(target)]
        detail = "MOV · ProRes 4444 + Alpha"
    elif output_format.startswith("WebM"):
        target = OUTPUT_DIR / f"{Path(source).stem}_cutout_vp9.webm"
        progress(0.92, desc="正在编码 VP9 + Alpha")
        # -auto-alt-ref 0 is required for VP9 alpha.  Note the alpha lives in a
        # side stream: ffmpeg's *native* vp9 decoder drops it, so downstream
        # tools must decode with -c:v libvpx-vp9 (browsers handle it natively).
        command = [FFMPEG, "-y", "-framerate", f"{fps(source):.8f}", "-i", sequence,
                   "-c:v", "libvpx-vp9", "-pix_fmt", "yuva420p", "-auto-alt-ref", "0",
                   "-crf", "24", "-b:v", "0", "-row-mt", "1", str(target)]
        detail = "WebM · VP9 + Alpha（下游解码需支持 VP9 alpha）"
    else:
        target = OUTPUT_DIR / f"{Path(source).stem}_cutout_frames.zip"
        progress(0.92, desc="正在打包 PNG 序列")
        with zipfile.ZipFile(target, "w", zipfile.ZIP_STORED) as zf:
            for file in made:
                zf.write(file, file.name)
        detail = "透明 PNG 序列 ZIP"
        command = None

    if command is not None:
        if Path(target).exists():
            Path(target).unlink()
        encoded = run(command)
        if encoded.returncode != 0 or not Path(target).exists():
            raise gr.Error("Alpha 视频编码失败：" + (encoded.stderr or "")[-700:])

    if not Path(target).exists():
        raise gr.Error("抠图输出文件未生成。")
    size_mb = Path(target).stat().st_size / 1_000_000
    progress(1, desc="完成 · ToonOut · BiRefNet · CUDA GPU")
    return str(target), (
        f"ToonOut · BiRefNet · CUDA GPU · 已处理 {len(files)} 帧，抠图用时 {mask_seconds:.1f} 秒，"
        f"输出 {detail}（{size_mb:.1f} MB）。"
    )


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


@lru_cache(maxsize=1)
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

    progress(0.05, desc=f"{label} 正在解码视频帧")
    extracted = run([FFMPEG, "-y", "-i", source, str(frames / "frame_%06d.png")])
    if extracted.returncode != 0:
        raise gr.Error("视频抽帧失败：" + extracted.stderr[-600:])

    total = len(list(frames.glob("*.png")))
    if not total:
        raise gr.Error("视频中没有可处理的帧。")

    progress(0.15, desc=f"{label} 正在超分 {total} 帧")
    command = [REALCUGAN, "-i", str(frames), "-o", str(out),
               "-s", str(scale), "-n", str(noise), "-m", str(directory)]
    if tta:
        command.append("-x")
    result = run(command)
    if result.returncode != 0:
        raise gr.Error(f"{label} 视频超分失败：" + result.stderr[-600:])
    made = sorted(out.glob("*.png"))
    if not made:
        raise gr.Error(f"{label} 没有输出任何帧。")

    progress(0.9, desc=f"{label} 正在编码视频")
    target = OUTPUT_DIR / f"{Path(source).stem}_realcugan{scale}x.mp4"
    # An Nx upscale multiplies pixel count by N^2, so scale the source bitrate
    # the same way and keep a floor for clean anime edges.
    target_bitrate = max(8_000_000, int((video_bitrate(source) or 8_000_000) * scale * scale * 1.15))
    encoded = run([FFMPEG, "-y", "-framerate", str(fps(source)), "-i", str(out / "frame_%06d.png"),
                   "-i", source, "-map", "0:v:0", "-map", "1:a?",
                   "-c:v", "libx264", "-crf", "12", "-preset", "slow", "-tune", "animation",
                   "-b:v", str(target_bitrate), "-maxrate", str(target_bitrate), "-bufsize", str(target_bitrate * 2),
                   "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", "-movflags", "+faststart", str(target)])
    if encoded.returncode != 0:
        raise gr.Error("视频编码失败：" + encoded.stderr[-600:])

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


def _depth_large_array(image):
    import numpy as np
    import torch
    from PIL import Image
    model, device = _depth_large_pipeline()
    resized = image.resize((518, 518), Image.Resampling.BICUBIC)
    array = np.asarray(resized, dtype=np.float32) / 255.0
    array = (array - np.asarray([0.485, 0.456, 0.406], dtype=np.float32)) / np.asarray([0.229, 0.224, 0.225], dtype=np.float32)
    inputs = {"pixel_values": torch.from_numpy(np.transpose(array, (2, 0, 1))).unsqueeze(0).to(device)}
    with torch.inference_mode():
        predicted = model(**inputs).predicted_depth
        predicted = torch.nn.functional.interpolate(
            predicted.unsqueeze(1), size=(image.height, image.width), mode="bicubic", align_corners=False
        )[0, 0]
    return predicted.float().cpu().numpy()


def _save_depth_array(depth, size, target, low=None, high=None):
    import numpy as np
    from PIL import Image
    if low is None or high is None:
        low, high = np.percentile(depth, [1, 99])
    normalized = np.clip((depth - low) / max(high - low, 1e-6) * 255, 0, 255).astype(np.uint8)
    Image.fromarray(normalized, mode="L").resize(size, Image.Resampling.BICUBIC).save(target)


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


def depth_video(video_path: str | None, progress=gr.Progress()):
    source = ensure_video(video_path)
    if not (DEPTH_LARGE_MODEL / "model.safetensors").exists():
        raise gr.Error(f"未找到 Depth Anything V2 Large 模型：{DEPTH_LARGE_MODEL}")
    try:
        from PIL import Image
    except ImportError as exc:
        raise gr.Error("深度图需要安装 Pillow。请重新运行 start.ps1。") from exc
    job = Path(tempfile.mkdtemp(prefix="depth_", dir=WORK_DIR))
    src_frames, dst_frames = job / "src", job / "dst"
    src_frames.mkdir(); dst_frames.mkdir()
    extract = run([FFMPEG, "-y", "-i", source, str(src_frames / "frame_%06d.png")])
    if extract.returncode != 0:
        raise gr.Error("视频抽帧失败：" + extract.stderr[-700:])
    files = sorted(src_frames.glob("*.png"))
    if not files:
        raise gr.Error("视频中没有可处理的帧。")
    try:
        progress(0.05, desc="Depth Anything V2 Large · 估算全片深度范围")
        low, high = _depth_global_range(files)
        for index, file in enumerate(files, 1):
            image = Image.open(file).convert("RGB")
            _save_depth_array(_depth_large_array(image), image.size, dst_frames / file.name, low, high)
            if index % 10 == 0 or index == len(files):
                progress(index / max(len(files), 1), desc=f"Depth Anything V2 Large · {index}/{len(files)} 帧")
        target = OUTPUT_DIR / f"{Path(source).stem}_depth.mp4"
        result = run([FFMPEG, "-y", "-framerate", str(fps(source)), "-i", str(dst_frames / "frame_%06d.png"), "-i", source,
                      "-map", "0:v:0", "-map", "1:a?", "-c:v", "libx264", "-crf", "10", "-preset", "slow",
                      "-pix_fmt", "yuv420p", "-c:a", "copy", "-shortest", str(target)])
        if result.returncode != 0:
            raise gr.Error("深度图视频编码失败：" + result.stderr[-700:])
        return str(target), (
            f"已处理 {len(files)} 帧，使用 Depth Anything V2 Large 输出灰度深度图视频"
            f"（全片统一深度范围 {low:.2f}–{high:.2f}，跨镜头灰阶一致）。"
        )
    except gr.Error:
        raise
    except Exception as exc:
        raise gr.Error(f"Depth Anything V2 Large 视频深度图失败：{exc}") from exc
    finally:
        shutil.rmtree(job, ignore_errors=True)



from types import SimpleNamespace
from ui import build_ui, CSS, THEME
from lifecycle import request_exit, watch_launcher

demo = build_ui(SimpleNamespace(**globals()))

if __name__ == "__main__":
    watch_launcher()
    demo.launch(inbrowser=False, server_name="127.0.0.1", server_port=7860, show_error=True,
                theme=THEME, css=CSS, footer_links=[])
