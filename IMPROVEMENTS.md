# Deep-Live-Cam Pro — 优化说明

基于 [Deep-Live-Cam](https://github.com/hacksider/deep-live-cam) AGPL v3 协议深度优化。

## 安装

1. 下载完整压缩包，解压到任意目录
2. 双击 `run-cuda.bat` 启动

> 无需安装 Python、pip、CUDA 等依赖，已内置 wzf311 便携 Python 环境。
> 仅支持 NVIDIA 显卡（CUDA 12.9），推荐 8GB 显存及以上（NVIDIA 显卡）。

## 速度优化

### CUDA Graph 内核重放
- 换脸模型 + GPEN-512 的 GPU 内核启动序列只录制一次，后续帧直接重放
- 消除每帧 CPU→GPU 内核调度延迟，**帧率更稳、延迟更低**

### 粘贴回写仅处理人脸裁剪区
- 官方对整帧做全图混合（1920×1080 = 2M 像素）
- 本版只在人脸 bbox 范围内运算（典型 200×200 = 40K 像素），**混合步骤快 ~50 倍**
- cv2 SIMD uint8 路径比 float32 往返快 **7-8 倍**

### 异步人脸检测
- 检测和换脸流水线并行，不互相等待

---

## 画质优化

### LAB 色彩迁移（GPEN-512 独有）
- 增强后的人脸通过 LAB 颜色空间统计匹配，**肤色与原始一致**，不会出现"增强脸偏白/偏黄"
- 官方版增强后人脸和周围皮肤有明显色差

### Edge Feather 边缘羽化（可调）
- 控制换脸边缘柔和度 0-100
- 同时作用于换脸 Swapper、GFPGAN-1024、GPEN-BFR-512 三个处理器

### Poisson Blend 泊松融合
- 蒙版从换脸仿射变换推导（而非独立 landmark 检测），**帧间零抖动**
- 解泊松方程做梯度域融合，边缘自然无痕

### 修复"外国人脸"幻觉
- 官方版 CUDA 环境下偶发生成不相关人脸，本版已彻底修复

### 修复黑色边框
- 官方版 GPEN-512 增强后脸上有可见方形/圆形边界，本版蒙版与变形区域精确对齐

---

## Bug 修复

| 问题 | 官方 | 本版 |
|------|------|------|
| 每次启动重下 529MB 模型 | 有 | 已修复 |
| GPEN-256 下载 404 死循环 | 有 | 已修复 |
| GFPGAN-1024 缺失直接崩溃 | 有 | 已修复 |
| CUDA 下偶现随机外国人脸 | 有 | 已修复 |
| GPEN-512 黑色方框/圆框 | 有 | 已修复 |

---

## 新增功能

- Edge Feather 滑块（边缘羽化 0-100）
- Poisson Blend 泊松融合开关
- 完整中文汉化（130+ 条文案）
- 去除 NSFW 审核 + TensorFlow 依赖
- 便携式打包，免安装运行

---

## 技术栈

| 组件 | 版本 |
|------|------|
| Python | 3.11 |
| ONNX Runtime | 1.26.0 |
| CUDA | 12.9 |
| cuDNN | 9.x |
| 执行提供器 | CUDA |
| 人脸检测 | insightface buffalo_l |
| 人脸增强 | GFPGAN-1024 / GPEN-BFR-512 |

---

## 协议

AGPL v3 — 基于 [Deep-Live-Cam](https://github.com/hacksider/deep-live-cam) 修改，保留原始版权声明。
