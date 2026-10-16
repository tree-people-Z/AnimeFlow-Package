"""AnimeFlow workspace layout; processing functions live in app.py."""
import gradio as gr

FONT = ['Segoe UI', 'Microsoft YaHei UI', 'Microsoft YaHei', 'PingFang SC', 'Noto Sans CJK SC', 'sans-serif']
THEME = gr.themes.Base(primary_hue="indigo", neutral_hue="slate", font=FONT,
                       font_mono=['Consolas', 'monospace'], radius_size="lg")
CSS = """
body { background: #f5f6f9; }
.gradio-container { width: 100% !important; box-sizing: border-box; max-width: 1240px !important; margin: 0 auto; padding: 24px 32px 32px !important; }
.gradio-container, .gradio-container input, .gradio-container textarea,
.gradio-container button { font-family: 'Segoe UI', 'Microsoft YaHei UI', 'Microsoft YaHei', 'PingFang SC', sans-serif !important; }
.gradio-container > .main { padding: 0 !important; }
#brand { display: flex; align-items: baseline; gap: 14px; padding: 0 0 18px; }
#brand h1 { margin: 0; font-size: 26px; font-weight: 650; letter-spacing: -.6px; }
#brand span { font-size: 13px; color: #778194; }
.workspace { gap: 20px !important; padding-top: 16px; }
.tool-body { gap: 24px !important; padding: 22px !important; border: 1px solid var(--border-color-primary); border-radius: 14px; background: var(--block-background-fill); }
.tool-controls { gap: 16px !important; }
.tool-result { gap: 12px !important; }
.tool-body > .tool-controls { flex-basis: 290px !important; }
.tool-body > .tool-result { flex-basis: 440px !important; }
.tool-nav > .tab-wrapper { gap: 4px; padding: 0 0 12px !important; border-bottom: 0 !important; flex-wrap: wrap !important; }
.tool-nav > .tab-wrapper [role="tab"] { padding: 9px 18px !important; border-radius: 8px !important; border: 1px solid transparent !important; font-size: 14px !important; white-space: nowrap; }
.tool-nav > .tab-wrapper [role="tab"].selected { color: var(--color-accent) !important; background: var(--background-fill-secondary) !important; border-color: var(--border-color-primary) !important; }
.tool-nav > .tabitem { padding: 0 !important; border: 0 !important; }
.model-note p { font-size: 12px !important; color: var(--body-text-color-subdued); margin: 0; }
.tip-note p { font-size: 12px !important; line-height: 1.55 !important; color: var(--body-text-color-subdued); margin: 0 !important; }
.tip-note { padding: 8px 10px !important; border-left: 3px solid var(--color-accent); background: var(--background-fill-secondary); border-radius: 6px; }
.run-button { min-height: 42px !important; border-radius: 9px !important; }
.status-box textarea { font-size: 13px !important; line-height: 1.6 !important; }
.gradio-container label { word-break: normal !important; overflow-wrap: normal !important; }
.gradio-container .block { min-width: 0; }
footer { display: none !important; }
@media (max-width: 760px) {
  .gradio-container { padding: 16px 14px !important; }
  #brand { padding-bottom: 12px; }
  #brand h1 { font-size: 23px; }
  .tool-body { padding: 16px !important; gap: 18px !important; }
  .tool-nav > .tab-wrapper [role="tab"] { padding: 8px 12px !important; }
  .tool-body > .tool-controls, .tool-body > .tool-result { flex-basis: 100% !important; min-width: 0 !important; }
}
"""


def status(label="处理状态"):
    return gr.Textbox(label=label, lines=2, max_lines=5, interactive=False, elem_classes="status-box")


def run_button(label):
    return gr.Button(label, variant="primary", elem_classes="run-button")


def build_ui(b):
    with gr.Blocks(title="AnimeFlow · 动画工作台") as demo:
        gr.HTML('<header id="brand"><h1>AnimeFlow</h1><span>动画工作台 · 本地处理 · 结果保存到 outputs</span></header>')
        with gr.Tabs(overflow_behavior="wrap"):
            with gr.Tab("视频", id="video"):
                with gr.Column(elem_classes="workspace"):
                    video = gr.Video(label="输入视频", sources=["upload"], height=210)
                    gr.Markdown("流程建议：动画通常先去重，再补帧；超分、抠图、深度图和抽帧可以直接处理原视频。",
                                elem_classes="tip-note")
                    with gr.Tabs(overflow_behavior="wrap", elem_classes="tool-nav"):
                        with gr.Tab("动画去重"):
                            with gr.Row(elem_classes="tool-body"):
                                with gr.Column(scale=2, min_width=280, elem_classes="tool-controls"):
                                    sensitivity = gr.Slider(.5, 4, value=1, step=.1, label="重复帧灵敏度",
                                                             info="1.0 适合多数动画；调高会更积极地合并近似画面。")
                                    output_format = gr.Radio([("MP4", "MP4（便于继续处理）"), ("MKV · 无损", "无损中间文件（MKV）")],
                                                             value="MP4（便于继续处理）", label="输出格式",
                                                             info="MP4 体积较小；MKV 使用 FFV1 无损编码，文件会很大。")
                                    gr.Markdown("去重会移除连续的重复/近重复帧并重建时间轴，不会自动补帧；完成后可在“补帧”里选择去重结果。",
                                                elem_classes="tip-note")
                                    dedup_btn = run_button("移除重复帧")
                                with gr.Column(scale=3, min_width=340, elem_classes="tool-result"):
                                    dedup_video = gr.Video(label="去重结果", height=260, interactive=False)
                                    dedup_status = status()
                        with gr.Tab("补帧"):
                            with gr.Row(elem_classes="tool-body"):
                                with gr.Column(scale=2, min_width=280, elem_classes="tool-controls"):
                                    gr.Markdown("GMFSS Fortuna · AnimeRun + RIFE 4.26-heavy 精修 · CUDA",
                                                elem_classes="model-note")
                                    source_choice = gr.Radio(["原视频", "去重结果"], value="原视频", label="处理来源",
                                                             info="可以直接选原视频；只有完成去重后才能选去重结果。")
                                    factor = gr.Radio([("2×", 2), ("3×", 3), ("4×", 4)], value=2, label="补帧倍数",
                                                      info="2× 最稳妥；4× 更流畅但耗时和显存明显增加。")
                                    gr.Markdown("需要 NVIDIA CUDA。模型或 CUDA 缺失时会直接报错，不会偷偷切换低质量模型；输出会保留原音轨并提高视频码率目标。",
                                                elem_classes="tip-note")
                                    interp_btn = run_button("开始补帧")
                                with gr.Column(scale=3, min_width=340, elem_classes="tool-result"):
                                    interpolated = gr.Video(label="补帧结果", height=260, interactive=False)
                                    interp_status = status()
                        with gr.Tab("超分"):
                            with gr.Row(elem_classes="tool-body"):
                                with gr.Column(scale=2, min_width=280, elem_classes="tool-controls"):
                                    up_model = gr.Dropdown(b.UPSCALE_CHOICES, value=b.UPSCALE_CHOICES[0], label="超分模型",
                                                            info="动漫线稿优先选 Pro 2×/3×；4× 使用 SE 权重。")
                                    up_denoise = gr.Dropdown(list(b.DENOISE_CHOICES), value=b.DEFAULT_DENOISE, label="降噪强度",
                                                             info="默认 -1 保留细节；只有片源噪点明显时再选强力降噪。")
                                    up_tta = gr.Checkbox(value=False, label="TTA 极致质量", info="约慢 4 倍，线条与纹理最干净")
                                    gr.Markdown("超分使用 Real-CUGAN Vulkan GPU。视频会重新编码并映射原音轨；源视频越大，处理时间和显存占用越高。",
                                                elem_classes="tip-note")
                                    up_btn = run_button("开始超分")
                                with gr.Column(scale=3, min_width=340, elem_classes="tool-result"):
                                    up_video = gr.Video(label="超分结果", height=260, interactive=False)
                                    up_status = status()
                        with gr.Tab("镜头检测"):
                            with gr.Row(elem_classes="tool-body"):
                                with gr.Column(scale=2, min_width=280, elem_classes="tool-controls"):
                                    gr.Markdown("TransNetV2 神经网络镜头检测", elem_classes="model-note")
                                    threshold = gr.Slider(.1, .9, value=.5, step=.05, label="切镜阈值",
                                                          info="转场概率，0.5 为官方默认，调低更敏感")
                                    gr.Markdown("输出可编辑的 TXT 镜头列表，包含切镜时间码、帧号和场景区间。",
                                                elem_classes="tip-note")
                                    scene_btn = run_button("检测镜头")
                                with gr.Column(scale=3, min_width=340, elem_classes="tool-result"):
                                    scene_file = gr.File(label="镜头列表", interactive=False)
                                    scene_status = status()
                        with gr.Tab("抠图"):
                            with gr.Row(elem_classes="tool-body"):
                                with gr.Column(scale=2, min_width=280, elem_classes="tool-controls"):
                                    gr.Markdown("ToonOut · BiRefNet · CUDA GPU · 1024px", elem_classes="model-note")
                                    cutout_format = gr.Dropdown(b.CUTOUT_OUTPUTS, value=b.CUTOUT_OUTPUTS[0],
                                                                label="输出格式",
                                                                info="PNG 序列最适合继续合成；MOV/WebM 可直接保留 Alpha。")
                                    gr.Markdown("视频抠图会逐帧推理，速度取决于片长和显卡；模型不完整或 CUDA 不可用时会直接提示。",
                                                elem_classes="tip-note")
                                    cutout_btn = run_button("导出透明帧")
                                with gr.Column(scale=3, min_width=340, elem_classes="tool-result"):
                                    cutout_file = gr.File(label="抠图结果", interactive=False)
                                    cutout_status = status()
                        with gr.Tab("深度图"):
                            with gr.Row(elem_classes="tool-body"):
                                with gr.Column(scale=2, min_width=280, elem_classes="tool-controls"):
                                    gr.Markdown("模型 · Depth Anything V2 Large　·　灰阶按全片统一范围映射",
                                                elem_classes="model-note")
                                    gr.Markdown("输出为灰度深度视频：白色表示更近，黑色表示更远。整段视频使用统一范围，镜头之间不会跳灰阶。",
                                                elem_classes="tip-note")
                                    depth_btn = run_button("生成深度图")
                                with gr.Column(scale=3, min_width=340, elem_classes="tool-result"):
                                    depth_result = gr.Video(label="深度图视频", height=260, interactive=False)
                                    depth_status = status()
                        with gr.Tab("导出帧"):
                            with gr.Row(elem_classes="tool-body"):
                                with gr.Column(scale=2, min_width=280, elem_classes="tool-controls"):
                                    every = gr.Slider(1, 60, value=1, step=1, label="帧间隔", info="1 表示导出全部帧；2 表示每隔 1 帧导出 1 张。")
                                    fmt = gr.Radio(["PNG（无损）", "JPG（体积小）"], value="PNG（无损）", label="图片格式",
                                                   info="动画后期或抠图建议 PNG；只预览时可选 JPG。")
                                    gr.Markdown("右侧会显示连续重复/近重复帧的预览，便于判断是否需要先做动画去重。",
                                                elem_classes="tip-note")
                                    extract_btn = run_button("导出图片")
                                with gr.Column(scale=3, min_width=340, elem_classes="tool-result"):
                                    frames_file = gr.File(label="图片序列 · ZIP", interactive=False)
                                    gallery = gr.Gallery(label="重复帧预览", columns=4, height=200, object_fit="contain")
                                    extract_status = status()

                def interpolate_selected(original, prepared, choice, multiple, progress=gr.Progress()):
                    if choice == "去重结果" and not prepared:
                        raise gr.Error("请先完成动画去重，或选择原视频。")
                    return b.interpolate(original, prepared if choice == "去重结果" else None, multiple,
                                         b.INTERP_ENGINES[0], progress)

                dedup_btn.click(b.deduplicate, [video, sensitivity, output_format], [dedup_video, dedup_status])
                interp_btn.click(interpolate_selected, [video, dedup_video, source_choice, factor], [interpolated, interp_status])
                up_btn.click(b.video_enhance, [video, up_model, up_denoise, up_tta], [up_video, up_status])
                scene_btn.click(b.detect_scenes, [video, threshold], [scene_file, scene_status])
                cutout_btn.click(b.video_cutout, [video, cutout_format], [cutout_file, cutout_status])
                depth_btn.click(b.depth_video, video, [depth_result, depth_status])
                extract_btn.click(b.extract_frames, [video, every, fmt], [frames_file, gallery, extract_status])
                video.change(lambda: None, outputs=dedup_video, queue=False)

            with gr.Tab("图片", id="image"):
                with gr.Column(elem_classes="workspace"):
                    with gr.Tabs(overflow_behavior="wrap", elem_classes="tool-nav"):
                        with gr.Tab("单张处理"):
                            with gr.Row(elem_classes="tool-body"):
                                with gr.Column(scale=2, min_width=280, elem_classes="tool-controls"):
                                    image_input = gr.File(label="导入图片", file_types=["image"], type="filepath", height=160)
                                    gr.Markdown("支持 PNG、JPG、WEBP 等常见图片格式。", elem_classes="tip-note")
                                    image_preview = gr.Image(label="原图", height=180, interactive=False, buttons=[])
                                    task = gr.Radio(["抠图", "超分", "重新打光", "深度图"], value="抠图", label="处理功能",
                                                    info="超分时才需要选择模型；其它功能使用固定高质量模型。")
                                    image_model = gr.Dropdown(b.UPSCALE_CHOICES, value=b.UPSCALE_CHOICES[0], label="超分模型", visible=False)
                                    gr.Markdown("抠图输出透明 PNG；超分使用 Real-CUGAN；重新打光使用 Marigold Normals；深度图使用 Depth Anything V2 Large。",
                                                elem_classes="tip-note")
                                    image_denoise = gr.Dropdown(list(b.DENOISE_CHOICES), value=b.DEFAULT_DENOISE, label="降噪强度", visible=False)
                                    image_tta = gr.Checkbox(value=False, label="TTA 极致质量", visible=False)
                                    image_run = run_button("开始处理")
                                with gr.Column(scale=3, min_width=340, elem_classes="tool-result"):
                                    image_output = gr.Image(label="处理结果", height=360, interactive=False, buttons=["download", "fullscreen"])
                                    image_status = status()
                        with gr.Tab("批量处理"):
                            with gr.Row(elem_classes="tool-body"):
                                with gr.Column(scale=2, min_width=280, elem_classes="tool-controls"):
                                    batch_input = gr.Files(label="导入图片", file_types=["image"], type="filepath")
                                    gr.Markdown("可一次选择多张，结果会打包为 ZIP。", elem_classes="tip-note")
                                    batch_task = gr.Radio(["抠图", "超分", "重新打光", "深度图"], value="抠图", label="处理功能")
                                    batch_model = gr.Dropdown(b.UPSCALE_CHOICES, value=b.UPSCALE_CHOICES[0], label="超分模型", visible=False)
                                    batch_denoise = gr.Dropdown(list(b.DENOISE_CHOICES), value=b.DEFAULT_DENOISE, label="降噪强度", visible=False)
                                    batch_tta = gr.Checkbox(value=False, label="TTA 极致质量", visible=False)
                                    gr.Markdown("批量任务会逐张处理；单张失败不会丢弃其它成功结果，状态栏会列出失败文件。",
                                                elem_classes="tip-note")
                                    batch_run = run_button("批量处理")
                                with gr.Column(scale=3, min_width=340, elem_classes="tool-result"):
                                    batch_output = gr.File(label="处理结果 · ZIP", interactive=False)
                                    batch_status = status()

                def upscale_fields(selected):
                    show = selected == "超分"
                    return (gr.update(visible=show), gr.update(visible=show), gr.update(visible=show))

                def image_process(path, selected, model, denoise, tta):
                    return b.image_stub(path, selected, model, denoise, tta)

                def batch_process(paths, selected, model, denoise, tta, progress=gr.Progress()):
                    return b.batch_images(paths, selected, model, denoise, tta, progress)

                image_input.change(lambda path: path, image_input, image_preview, queue=False)
                task.change(upscale_fields, task, [image_model, image_denoise, image_tta], queue=False)
                batch_task.change(upscale_fields, batch_task, [batch_model, batch_denoise, batch_tta], queue=False)
                image_run.click(image_process, [image_input, task, image_model, image_denoise, image_tta], [image_output, image_status])
                batch_run.click(batch_process, [batch_input, batch_task, batch_model, batch_denoise, batch_tta], [batch_output, batch_status])
            with gr.Tab("模型与环境", id="settings"):
                with gr.Column(elem_classes="workspace"):
                    gr.Markdown("### 模型与运行环境\n下面显示的是整合包实际使用的模型和状态。处理失败时先看这里的配置状态。")
                    gr.Markdown(b.model_catalog())
                    gr.Markdown(b.backend_status(), elem_classes="model-note")
                    gr.Markdown("说明：补帧使用的是 GMFSS Fortuna 主干与 RIFE 4.26-heavy 精修分支的组合权重；它不是可单独切换的 RIFE 4.25 模型。带 CUDA 的功能需要 NVIDIA 驱动，缺失时会直接报错，不会自动回退。",
                                elem_classes="tip-note")
                    exit_status = gr.Textbox(label="运行状态", interactive=False)
                    exit_btn = gr.Button("退出应用", variant="stop")
                    exit_btn.click(b.request_exit, outputs=exit_status, queue=False)
    return demo
