# AnimeFlow

面向动画视频的本地 AI 工具包：抽帧、补帧、超分、深度图、抠图、重新打光。
当前版本：**1.0.0**

## 整合包下载

完整整合包（包含内置运行时与模型）可通过夸克网盘获取：

- [夸克网盘下载 AnimeFlow-Package](https://pan.quark.cn/s/5e541cf6296c)
- 分享码：`/~ca3c3bUFEb~/`


## 快速启动

双击 `start.bat`（或 `启动 AnimeFlow.bat`）即可。**不需要安装 Python，也不联网下载依赖**，解释器和全部依赖都在包里。
启动日志保存在 `work/startup.log`。
启动器会等待本地服务就绪后自动打开浏览器；如果浏览器没有弹出，可手动访问 `http://127.0.0.1:7860`。
启动时会检查 GitHub Release 是否有新版本；网络不可用时会跳过检查并照常启动。

重复启动会打开已有的 AnimeFlow 页面。退出可使用 `stop.bat`（或 `停止 AnimeFlow.bat`）或“模型与环境”页的“退出应用”。
关闭主启动窗口会结束应用及其子任务；仅关闭浏览器页面不会退出后台。

**前置条件**：64 位 Windows + NVIDIA 显卡驱动（CUDA 功能需要；FFmpeg 已随包提供）。


## 功能

### 视频
- **抽帧**：移除重复帧（动画优化）、连续抽帧、导出图片序列
- **补帧**：GMFSS Fortuna + **RIFE 4.26-heavy 精修**（CUDA）。支持 2×/3×/4×
- **超分**：Real-CUGAN（默认 Pro 2×，另有 Pro 3× / 4×），可选降噪强度与 TTA 极致质量
- **镜头检测**：TransNetV2 神经网络
- **深度图**：Depth Anything V2 Large
- **抠图**：ToonOut · BiRefNet 动漫抠图，输出透明 PNG 序列 / ProRes 4444 MOV / VP9 WebM

### 图片
- **超分**：与视频同款 Real-CUGAN
- **深度图**：Depth Anything V2 Large
- **抠图**：ToonOut · BiRefNet 动漫抠图
- **重新打光**：Marigold Normals 高质量法线

## 模型

整合包已配置动漫专用高质量模型，无需额外下载。模型清单见 `models\MANIFEST.json`。
**每个功能只保留实测最优的一个选项**，不做并列：

- **GMFSS Fortuna AnimeRun + RIFE 4.26-heavy**：动画补帧。RIFE 4.26 替换了 GMFSS 原配的旧 IFNet，实测平均 **+0.67 dB**（6 个用例中 4 个更好）
- **Real-CUGAN**：动漫超分。Pro 2×/3× 是上游真正打磨过的权重；4× 只有 SE 版（作者认为 Pro 4× 效果不达标，未放出）
- **TransNetV2**：神经网络镜头边界检测，对动画硬切与快速运动都稳定
- **Depth Anything V2 Large**：高质量深度图与重新打光基础
- **ToonOut · BiRefNet**：动漫抠图，Swin-L 主干，1024px 推理
- **Marigold Normals**：表面法线与重新打光

被移除的并列选项（均因实测更差）：纯 GMFSS Fortuna、RIFE Anime 2022、DRBA 逐像素时点校正、Real-ESRGAN AnimeVideoV3、FFmpeg 场景分数、逐帧自适应深度灰阶。
具体数据见 `CONFIG.md`。

## 抠图方案

图片与视频抠图使用**同一个模型**（ToonOut · BiRefNet），区别在输入/输出形态：

| | 图片抠图 | 视频抠图 |
|---|---|---|
| 模型 | ToonOut · BiRefNet（Swin-L，885 MB） | 同 |
| 运行时 | PyTorch CUDA fp16，独立子进程隔离 | 同 |
| 输入 | 单张图片 | FFmpeg 管道解码为 RGB 帧，不落地中间 PNG |
| 推理 | 缩放到 1024×1024 | 逐帧同样处理 |
| 输出 | **透明 PNG**（alpha 按原尺寸回缩） | 三种容器，见下 |
| 速度 | 约 0.2 秒/帧（首次含加载约 30 秒） | 逐帧约 0.2 秒 + 解码/打包 |
| 显存 | 约 1.6 GiB | 同 |

视频抠图输出格式（均实测 alpha 可正确还原）：

| 容器 | 说明 |
|------|------|
| 透明 PNG 序列（ZIP） | 无损，任何工具都能读 |
| MOV · ProRes 4444 + Alpha | 剪辑软件通用 |
| WebM · VP9 + Alpha | 体积最小（同一段素材约 1/37），浏览器直接播 |

> VP9 的 alpha 存在旁路流里，FFmpeg 原生 vp9 解码器会**静默丢掉 alpha**，下游需用 `-c:v libvpx-vp9` 解码。

## 画质取向

本工具按“动漫画质优先，可接受较慢”取舍：

- 补帧使用 **RIFE 4.26 精修**：实测比 GMFSS 原配 IFNet 高 **+0.67 dB**
- 超分默认 **Real-CUGAN Pro 2×**：1080p 源 → 干净 4K，且使用的是上游真正打磨过的权重
- 降噪默认 **保守 -1**：保留细节，不涂抹线条
- 可勾选 **TTA 极致质量**：约慢 4 倍，换来最干净的线条与纹理
- 超分整段视频一次性交给 Vulkan 运行器处理（而非逐帧起进程），实测约快 2.5 倍

## 设计

简洁、优雅、直观，主流程清晰可见，高级选项收起。批量处理自动打包 ZIP。所有输出在 `outputs/`，临时文件在 `work/`。

## 使用提示

- 动画建议先运行“动画去重”，确认重复帧预览后，再在“补帧”中选择“去重结果”；补帧也可以直接处理原视频。
- 补帧和视频抠图需要 NVIDIA CUDA；缺少 CUDA 或模型时会直接报错，不会偷偷切换到低质量后端。
- 补帧、超分会保留原音轨，并按生成后的帧数提高视频码率目标；抠图输出透明素材，不包含音频。
- 4×补帧、TTA 和 Marigold 重打光会明显变慢，属于质量优先模式的正常现象。
- 图片/视频处理结果写入 `outputs/`；成功任务会自动清理中间帧，任务中断后仍可以删除 `work/` 中的临时目录。

## 技术

本地 Python + PyTorch + ncnn Vulkan + FFmpeg。内置解释器和依赖位于 `runtime/python-embed/`。高质量模型优先使用 CUDA；不支持 CUDA 的功能会直接提示。

抠图与镜头检测使用 PyTorch CUDA 路径（`runtime/python-embed` 提供 timm / torchvision / kornia）。抠图的 GPU 推理在独立子进程执行，底层崩溃不会带走界面。

## 项目结构

```
AnimeFlow-Package/
├── start.bat / 启动 AnimeFlow.bat   # 双击启动（推荐）
├── stop.bat  / 停止 AnimeFlow.bat   # 停止后台
├── start.ps1 / stop.ps1             # PowerShell 启动脚本
├── app.py             # 主应用
├── gpu_cutout.py      # ToonOut 抠图（PyTorch CUDA）
├── scene_detect.py    # TransNetV2 镜头检测
├── vfi_plus.py        # GMFSS Fortuna + RIFE 4.26 精修补帧
├── ui.py              # 界面布局
├── launcher.py        # 启动与日志
├── lifecycle.py       # 进程生命周期
├── bin/               # FFmpeg / FFprobe、Real-CUGAN Vulkan 运行器
├── models/            # 模型权重（约 4.6 GB）
├── vendor/            # 内嵌第三方代码（drba、BiRefNet、TransNetV2）
└── runtime/
    └── python-embed/  # 内置 Python 3.13.15 解释器 + 全部依赖（约 4.9 GB）
```

## 便携性

这是一个**开箱即用的整合包**：解释器和全部 Python 依赖（PyTorch CUDA、Gradio、timm、torchvision、kornia 等）
都打包在 `runtime/python-embed/` 里。

- **目标机器不需要安装 Python、不需要 pip、不需要联网**
- 整个 `AnimeFlow-Package` 目录可以拷到任意路径、任意 Windows 电脑（x64），双击 `start.bat` 即用
- 启动脚本优先使用内置解释器；`.venv` 与系统 Python 只是开发时的回退选项
- 包内置解释器优先使用；`.venv` 与系统 Python 仅作为开发环境的备用路径

唯一的外部前提是 **NVIDIA 显卡驱动**（补帧与抠图需要 CUDA；显卡驱动由用户自行安装）。


