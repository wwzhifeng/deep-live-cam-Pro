<h1 align="center">Deep-Live-Cam Pro</h1>

<p align="center">
  Real-time face swap — one click, one image.
</p>

<p align="center">
  <video src="https://github.com/user-attachments/assets/97415ded-a36d-4b93-8d17-b40ec0fafc61" width="800" autoplay loop muted playsinline controls>
  </video>
</p>

<p align="center">
  <a href="https://pan.quark.cn/s/e41399439d45"><b>⬇️ 一键整合包下载</b></a>
</p>

##  Disclaimer

This software is designed for the AI-generated media industry — assisting artists, content creators, and character animation.

Users must obtain consent before using a real person's face and clearly label any output as a deepfake when sharing online. We are not responsible for end-user actions.

## 本地部署

### 方式一：便携版（推荐）

1. 下载完整压缩包并解压
2. 双击 `run-cuda.bat`

> 已内置 Python 3.11 + CUDA 12.9 + 全部依赖 + 所有模型，无需安装任何东西。
> 仅支持 NVIDIA 显卡，推荐 8GB 显存及以上（NVIDIA 显卡）。

### 方式二：从源码安装（进阶用户）

**前置条件：** 系统已安装 CUDA Toolkit 12.x、cuDNN 9.x、TensorRT。

```bash
git clone https://github.com/wwzhifeng/deep-live-cam-Pro.git
cd deep-live-cam-Pro
python -m venv venv
venv\Scripts\activate

# 先装 PyTorch（CUDA 12.9）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# 再装其余依赖
pip install -r requirements.txt
```

将以下模型放入 `models/` 目录（共 ~1.1GB）：
- `inswapper_128.onnx` — [下载](https://huggingface.co/hacksider/deep-live-cam/resolve/main/inswapper_128.onnx)（~529MB）
- `gfpgan-1024.onnx` — [下载](https://huggingface.co/hacksider/deep-live-cam/resolve/main/gfpgan-1024.onnx)（~349MB）
- `GPEN-BFR-512.onnx` — [下载](https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/GPEN-BFR/GPEN-BFR-512.onnx)（~272MB）

```bash
python run.py --execution-provider cuda
```

## Features

- Real-time webcam face swap
- Image and video processing
- Face enhancement (GFPGAN-1024 / GPEN-BFR-512)
- Edge feathering (adjustable 0-100)
- Poisson blend for stable, jitter-free output
- LAB color correction (natural skin tone matching)
- Full Chinese localization

## Improvements over upstream

See [IMPROVEMENTS.md](IMPROVEMENTS.md) for the full comparison.

- CUDA Graph kernel replay — smoother frame rate
- Paste-back limited to face bbox — 50x faster blending
- Fixed: black borders, foreign-face hallucination, repeated model downloads
- Chinese UI (130+ strings)

## License

AGPL v3 — based on [Deep-Live-Cam](https://github.com/hacksider/deep-live-cam).

## Credits

- [deepinsight](https://github.com/deepinsight) — insightface models
- [hacksider](https://github.com/hacksider/deep-live-cam) — original project
- [ffmpeg](https://ffmpeg.org/) — video processing

## 交流群

QQ 群：**773608333**（AI工具2群）

## Links

- [https://wangzhifeng.vip/](https://wangzhifeng.vip/)
