# AnimeFlow 动漫整合包 — 配置与方案

## 画质取向

**动画效果优先，可以接受较慢。** 默认档位选择上游真正打磨过的权重，并为最高画质保留可选开关（TTA、降噪强度）。

## 模型配置

| 模型 | 用途 | 大小 | 说明 |
|------|------|------|------|
| Real-CUGAN Pro | 动漫超分主力 | 14.8 MB | **2×/3×**，降噪修正、线条更干净、纹理涂抹更少 |
| Real-CUGAN SE | 动漫超分 4× | （同目录集） | **4× 仅此一套**，Pro 未发布 4× |
| GMFSS Fortuna + RIFE 4.26 | 动画补帧 | 71.6 MB | GMFSS union 权重 + RIFE 4.26-heavy 精修网络，CUDA |
| TransNetV2 | 镜头边界检测 | 29.1 MB | 神经网络，动画硬切/快速运动稳定 |
| Depth Anything V2 Large | 深度图 | 1279 MB | 图片/视频深度图 |
| ToonOut · BiRefNet | 动漫抠图 | 844 MB | PyTorch CUDA，Swin-L，1024px |
| Marigold Normals | 重新打光 | 2465 MB | 表面法线，8 DDIM steps |

**模型合计**：约 4.5 GB（仅 6 个模型，每个功能一个）

### 每个功能只保留一个最优选项

按实测结果做了收敛，同一功能不再并列多个模型：

| 功能 | 保留 | 移除（及原因） |
|------|------|---------------|
| 补帧 | GMFSS Fortuna + RIFE 4.26-heavy | 纯 GMFSS（同权重+旧 IFNet，更差）；RIFE Anime 2022（ncnn 引擎冻结在 2022，且需单独的 57 MB 权重）；DRBA 逐像素 DRM（平均 −0.41 dB） |
| 超分 | Real-CUGAN（2×/3×/4×） | Real-ESRGAN AnimeVideoV3（动漫上过锐、会改画风） |
| 镜头检测 | TransNetV2 | FFmpeg 场景分数（像素差值启发式，动画硬切易漏、快速平移易误检） |
| 深度图灰阶 | 全片统一范围 | 逐帧自适应（会让同一深度在不同镜头映射成不同灰阶，画面「呼吸」） |

倍率（2×/3×/4×）与降噪档位是**参数**，不算并列模型，予以保留。

### 抠图方案（图片 / 视频）

两者用**同一个模型**（ToonOut · BiRefNet），差别只在输入输出形态：

| | 图片抠图 | 视频抠图 |
|---|---|---|
| 运行时 | PyTorch CUDA fp16，独立子进程隔离 | 同 |
| 输入 | 单张图片 | FFmpeg 全解码为 PNG 序列 |
| 输出 | **透明 PNG** | 见下表三种容器 |
| 速度 | 约 0.2 秒/帧（首次含加载约 30 秒） | 逐帧约 0.2 秒 + 解码/打包 |
| 显存 | 约 1.6 GiB | 同 |

视频抠图的三种输出容器（都已实测 alpha 可正确还原）：

| 容器 | 同一段素材体积 | 说明 |
|------|--------------|------|
| 透明 PNG 序列（ZIP） | 677 KB | 无损，任何工具都能读 |
| MOV · ProRes 4444 + Alpha | 566 KB | 剪辑软件通用 |
| WebM · VP9 + Alpha | **18 KB** | 体积最小（约 1/37），浏览器直接播 |

> **注意**：VP9 的 alpha 存在旁路流里，FFmpeg 的**原生 vp9 解码器会静默丢掉 alpha**，下游必须用 `-c:v libvpx-vp9` 解码（浏览器无此问题）。ProRes 4444 与 PNG 序列没有这个坑。

### 关于 Real-CUGAN Pro 没有 4×

上游作者在 2022-05 的 Real-CUGAN-Pro 更新中说明：*由于对 4× 新版模型效果不满意，暂时只放出 2× 和 3×*。
因此 `models-pro` 只有 2×/3× 是**完整的官方集合**，不是下载不全。4× 必须使用 `models-se`。

Pro 相对 SE 的实际改进（来自上游 changelog）：

- 修掉了 RealESRGAN 式 USM 带来的**线条锐化过冲/振铃**
- GAN 训练时用前景/背景掩码抑制背景区域的对抗监督，**减少纹理涂抹**
- **不再放大压缩块噪点**

### 已移除的模型与运行库

- `anime-interp-rife-v4-26-large`：ncnn 运行器不支持，已删除
- `models/anime-cutout-birefnet-toonout-fp16/birefnet-toonout-fp16.onnx`（**469.6 MB**）：已删除。GPU 必崩（见下文），
  已被 PyTorch 版 `models/toonout-pytorch/birefnet_finetuned_toonout.pth` 完全取代
- `runtime/onnx-gpu/`（**412.8 MB**）：已删除。抠图改用 PyTorch 后，全工程再无任何代码 `import onnxruntime`；
  `requirements.txt` 与 `start.ps1` 的相应条目也已移除

本轮共释放 **882 MB**。

## 功能清单

### 视频
1. **动画去重**：移除重复帧，动画优化；可输出 MP4 或无损 MKV
2. **补帧**：GMFSS Fortuna + **RIFE 4.26-heavy 精修**（唯一补帧引擎，CUDA），支持 2×/3×/4×；可直接处理原视频，不需要先去重
3. **超分**：Real-CUGAN Pro 2×/3×、Real-CUGAN 4×，可选降噪强度与 TTA
4. **镜头检测**：TransNetV2 神经网络
5. **抠图**：ToonOut · BiRefNet，导出透明 PNG 序列 ZIP
6. **深度图**：Depth Anything V2 Large，全片统一灰阶范围
7. **导出帧**：PNG / JPG 序列 ZIP + 重复帧预览

### 图片
1. **超分**：同视频（含降噪强度与 TTA）
2. **深度图**：Depth Anything V2 Large
3. **抠图**：ToonOut · BiRefNet
4. **重新打光**：Marigold Normals

## 关键技术决策

### 为什么放弃 ONNX 抠图

原实现用 `onnxruntime-gpu` 跑 `birefnet-toonout-fp16.onnx`。在 RTX 5060（Blackwell **sm_120**）上实测：

| 后端 | 建会话 | 推理 |
|------|--------|------|
| CPU | 成功 | 成功，但 **7.9 秒/帧** |
| CUDA（4 组 session 选项全试过） | 成功 | **0xC0000005 访问冲突，进程崩溃** |

根因是官方 `onnxruntime-gpu` 预编译包不含 sm_120 内核。社区有重编版本，但它是 **cp312 + CUDA 13.1**，与本环境的 Python 3.13 + CUDA 12.8 不兼容。

改用 **PyTorch** 跑同一套权重（`joelseytre/toonout` 的 `birefnet_finetuned_toonout.pth`）：

- **0.20 秒/帧**（比 CPU ONNX 快约 39 倍）
- 显存峰值 **1.58 GiB**
- 与 `vendor/toonout_birefnet.py` 里的 `BiRefNet` 结构**完全匹配**（754 项权重，0 缺失 / 0 多余）

**注意**：该 checkpoint 的键名被包了两层前缀（`module._orig_mod.bb...`，DataParallel + torch.compile），必须循环剥离到稳定为止，否则 `load_state_dict` 会静默跳过全部权重、模型退化成随机输出。

**该 ONNX 文件与 `runtime/onnx-gpu` 均已删除**（详见上文「已移除的模型与运行库」）。

### 为什么超分改成整目录处理

原实现**每帧启动一个 `realcugan-ncnn-vulkan.exe` 进程**，每次都重新加载模型。改成把整个帧目录一次性交给运行器，由其内部的 load/proc/save 流水线并行处理。1920×1080 实测：

| 方式 | 秒/帧 |
|------|-------|
| 逐帧起进程（原） | 3.069 |
| **整目录（现）** | **1.245** |
| 整目录 + TTA | 5.069 |
| Pro 2× 整目录 | 0.698 |

即 **2.5 倍提速**，画质不变。

### 为什么镜头检测换成 TransNetV2

FFmpeg 的 `select='gt(scene,·)'` 是像素差值启发式：动画里两个明暗相近的赛璐璐之间硬切容易漏检，而快速平移又容易误检。TransNetV2 逐帧判别转场，权重（29 MB）已随包提供。

帧对齐已实测校准：真实切点在帧 8 时，模型标记转场帧 7、新镜头起始帧 8；报告输出的是**新镜头首帧**的时间戳（误差 0 帧）。

### 为什么补帧引入了 RIFE 4.26 精修网络（换的是网络，不是方案）

**结论先说：模型层面没有比 GMFSS Fortuna 更好的开源选择。** 调研了 RIFE 4.22/4.25/4.26、SGM-VFI、EMA-VFI、IFRNet、XVFI、FILM、ToonComposer 等，
在 8 GB / sm_120 上**没有开源模型能明显超过 GMFSS Fortuna AnimeRun 的动画质量**
（ToonComposer 自述需约 57 GB 显存；FlashVSR v1.1 权重约 6.9 GB 且作者未验证 RTX 50 系）。

但 **GMFSS Fortuna 的 union 权重可以换配一个更好的精修网络（IFNet）**。官方 release 配的是较旧的 rife 分支；
DRBA 项目（routineLife1/DRBA，MIT，SVFI 团队赞助）配的是 **RIFE 4.26-heavy**——Practical-RIFE 作者称 4.26 对动画场景有显著改善。

**客观验证**：用真实 1920×1080 动画帧做平移裁切生成已知真值（真值 = 时间中点），
2 个素材 × 3 种运动模式 = 6 个用例。变体：
A = 原版 GMFSS，B = GMFSS + RIFE 4.26（标量时间步），C = B + DRBA 逐像素时点。

| 效果 | 平均 | 更好的用例数 |
|------|------|-------------|
| **IFNet 升级（B−A）** | **+0.67 dB** | **4/6** |
| DRM 时点校正（C−B） | −0.41 dB | 2/6 |
| 两者合计（C−A） | +0.25 dB | — |

逐用例（素材 A / 素材 B）：

| 运动模式 | IFNet(B−A) | DRM(C−B) |
|----------|-----------|----------|
| 匀速平移 | +1.24 / +1.48 | −0.46 / −0.62 |
| 加速平移 | +1.15 / +1.27 | −2.27 / −2.35 |
| 双轴不同加速 | −0.64 / −0.49 | **+1.56 / +1.65** |

**结论：RIFE 4.26 精修网络是可靠的改进（默认开启）；DRBA 的逐像素时点校正平均为负，因此没有放进整合包界面。**

> 修正：第二轮曾基于「合成色块 + 单一运动」测试宣称 DRBA「4/4 全胜」。那个测试没有把 IFNet 的贡献与 DRM 分离，
> 结论不成立。上表是分离后的结果，并已在两份不同素材上复现。

**速度**（1920×1080，单输入帧）：原版 GMFSS 0.73 秒；+RIFE 4.26 版本跳过 DRM 计算，是增强路径里最快的；再开 DRBA 约增加 1.5 倍。

**接入要点**：

- 上游包名就叫 `models`，会与本工具的 `models/` 权重目录冲突，故重命名为 `drba_core`
- 上游在 **BGR** 色彩序下推理（OpenCV 习惯），本模块保持一致
- 上游 `-t/--times` 调度会产生**非标准帧序**（不保留中间原始帧），已改用等价于其 `-fps`/TMapper 的标准 N× 帧序
- 帧是流式处理的（3 帧前瞻），不会把整段视频读进内存
- 帧数严格等于 `输入帧数 × 倍数`，与其它补帧引擎一致，保证时长正确
- 场景切换处自动退回普通两帧路径，避免跨切镜做光流

### 为什么 RIFE 停留在 2022 版

`nihui/rife-ncnn-vulkan` 上游已冻结，最后一个 release 是 **20221029**，其内置模型只到 `rife-v4.6`。本地的 v4.22 / v4.25 / v4.26 ncnn 权重在实测中全部崩溃（`0xC0000005`），因为引擎版本早于模型格式。

排查过的替代方案与结论：

| 尝试 | 结果 |
|------|------|
| 换用更新的 ncnn 运行器 | 上游无更新，release 停在 20221029 |
| `-s 0.25` / `-s 0.75` 在目录模式下直接生成单代中间帧 | **失败**，rc=`0xFFFFFFFF`；`-s` 只对单对 `-0`/`-1` 模式有效 |
| 现有 4× 两遍 2× 链 | **正确**：4 帧 → 8 帧 → 16 帧，符合预期 |

目录模式的实际输出规律（实测）：N 帧输入 → **2N 帧**输出，排布为
`[f1, mid(f1,f2), f2, mid(f2,f3), …, fN, fN]`（末帧重复一次填充）。
按 `源帧率 × 倍数` 编码后总时长正确。

动画补帧的主力本来就是 CUDA 上的 **GMFSS Fortuna AnimeRun**（动画微调），RIFE 只是备选；
在没有可用新引擎的前提下，保持现状比引入半成品更稳妥。

## 用户界面

- 超分提供明确预设：`Real-CUGAN Pro 2×（动漫推荐）`、`Real-CUGAN Pro 3×`、`Real-CUGAN 4×（SE 权重）`
- 降噪强度：`保守 -1（保留细节 · 推荐）`、`不降噪 0`、`强力降噪 3（片源噪点多）`
- `TTA 极致质量` 复选框：约慢 4 倍
- 补帧使用唯一的 `GMFSS Fortuna + RIFE 4.26-heavy` 引擎，界面不提供未经整合的 RIFE 4.25/4.26 ncnn 选项
- 无静默 fallback：实际使用的模型名称会写进结果消息
- 输出消息明确标注模型名、缩放倍率与降噪档位

## 使用提示

- **动画推荐流程**：先运行“动画去重”，确认右侧重复帧预览，再在“补帧”中选择“去重结果”。补帧也可以直接选择原视频。
- **质量与速度**：补帧和视频抠图使用 CUDA，4×补帧、TTA 和 Marigold 重打光会明显变慢；这是质量换时间的正常表现。
- **音频与码率**：补帧和超分会映射原音轨并按生成后的帧数提高视频码率目标；抠图输出透明素材，不包含原音轨。
- **输出位置**：所有结果写入整合包的 `outputs`，临时帧写入 `work`；任务中断后可安全删除 `work` 下的临时目录。
- **失败处理**：本包不自动切换到低质量后端。若设置页显示模型缺失、CUDA 不可用或 FFmpeg 缺失，请先修复环境再重试。

## 技术栈

- **内置 Python 3.13.15 嵌入式解释器**（`runtime/python-embed/`），目标机器无需安装 Python
- PyTorch CUDA 12.8（`torch 2.11.0+cu128`，支持 Blackwell sm_120）；必须锁这个版本，因为 torchvision 0.26 硬性要求 `torch==2.11.0`
- ncnn Vulkan（Real-CUGAN）
- FFmpeg（随包提供）
- Gradio 6.30+（UI）
- 不再依赖 onnxruntime

## 便携性（整合包）

所有第三方包都在 `runtime/python-embed/Lib/site-packages`，由 `runtime/python-embed/python313._pth` 指定搜索路径：

```
python313.zip
.
Lib\site-packages
import site
```

启动脚本按此顺序挑解释器：

1. `runtime\python-embed\python.exe` —— 整合包内置，**正常情况走这个**
2. `.venv\Scripts\python.exe` —— 开发用虚拟环境（发布包中不存在）
3. `py -3` —— 系统 Python，最后的回退

**踩过的三个坑**（都已修，改代码时别踩回去）：

- 嵌入式发行版的 `._pth` 会**接管 sys.path，且不会把脚本自身目录加进去**。
  所以 `launcher.py` / `app.py` / `gpu_cutout.py` / `scene_detect.py` / `vfi_plus.py`
  都在导入本地模块前显式把自己的目录插进 `sys.path`；否则报 `ModuleNotFoundError: No module named 'lifecycle'`。
- `start.ps1` 含中文，**必须带 UTF-8 BOM**。Windows PowerShell 5.1 对无 BOM 文件按 ANSI 解码，
  会把中文变成乱码并破坏语法（曾报 `Unexpected token 'py' in expression or statement`）。
  新增/修改 `.ps1` 后请确认 BOM 仍在。
- 早期版本自带 `.venv` 里是 **CPU 版 torch**，且 `start.ps1` 优先用系统 Python，
  导致整合包实际上并不自包含。现已改为内置解释器优先。

**实测**：把整个 `AnimeFlow-Package` 目录拷到另一路径、且不带 `.venv`，
仍能启动并跑完全部 16 项功能测试。

## 启动

双击 `start.bat`（或 `启动 AnimeFlow.bat`）。

## 输出

- 所有结果输出到 `outputs/`
- 临时文件在 `work/`
- 批量图片处理自动打包 ZIP

## 注意

- 补帧引擎需要 CUDA；未检测到 CUDA 时直接报错，不自动回退
- 抠图在独立子进程执行 GPU 推理，底层崩溃不会带走界面
- 启动脚本会校验本地依赖，不会联网安装
