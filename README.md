<h1 align="center">Deep-Live-Cam Pro</h1>

<p align="center">
  Real-time face swap — one click, one image.
</p>

<p align="center">
  <img src="media/demo.gif" alt="Demo GIF" width="800">
</p>

##  Disclaimer

This software is designed for the AI-generated media industry — assisting artists, content creators, and character animation.

Users must obtain consent before using a real person's face and clearly label any output as a deepfake when sharing online. We are not responsible for end-user actions.

## Quick Start (Portable)

1. Download and extract the zip
2. Double-click `run-cuda.bat`

> **Requirements:** NVIDIA GPU (RTX 3060+ recommended), bundled Python environment included — nothing to install.

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

## Links

- [https://wangzhifeng.vip/](https://wangzhifeng.vip/)
