"""GPEN-BFR-512 face enhancer — ONNX-based face restoration at 512x512.

Includes CUDA Graph support for Blackwell/Lovelace GPUs — the 512×512 input
is fixed-size so the GPU kernel launch sequence is recorded once and replayed
with near-zero CPU overhead on every subsequent frame.
"""

from typing import Any, List
import os
import threading

import cv2
import numpy as np

import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.typing import Frame, Face
from modules.utilities import (
    is_image,
    is_video,
)
from modules.processors.frame._onnx_enhancer import (
    create_onnx_session,
    warmup_session,
)

NAME = "DLC.FACE-ENHANCER-GPEN512"
INPUT_SIZE = 512
MODEL_URL = "https://github.com/harisreedhar/Face-Upscalers-ONNX/releases/download/GPEN-BFR/GPEN-BFR-512.onnx"
MODEL_FILE = "GPEN-BFR-512.onnx"

ENHANCER = None
THREAD_LOCK = threading.Lock()
_CUDA_GRAPH_ENABLED = False

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)


def _init_cuda_graph(session):
    """Wrap session with CUDA Graph replay for the GPEN-512 model.

    The model has fixed input 1×3×512×512, so the kernel launch sequence
    can be recorded once and replayed with minimal CPU overhead.
    """
    global _CUDA_GRAPH_ENABLED
    import onnxruntime as ort
    try:
        providers = [('CUDAExecutionProvider', {'enable_cuda_graph': '1'})]
        cg_sess = ort.InferenceSession(
            os.path.join(models_dir, MODEL_FILE), providers=providers,
        )

        inp = np.zeros((1, 3, INPUT_SIZE, INPUT_SIZE), dtype=np.float32)
        ort_inp = ort.OrtValue.ortvalue_from_numpy(inp, 'cuda', 0)
        io = cg_sess.io_binding()
        io.bind_ortvalue_input(cg_sess.get_inputs()[0].name, ort_inp)
        io.bind_output(cg_sess.get_outputs()[0].name, 'cuda', 0)

        cg_sess.run_with_iobinding(io)  # records the graph

        session._cg_sess = cg_sess
        session._cg_io = io
        session._cg_ort_inp = ort_inp
        session._cg_recorded = True
        _CUDA_GRAPH_ENABLED = True
        print(f"[{NAME}] CUDA Graph session initialized — GPU kernel replay active")
    except Exception as e:
        print(f"[{NAME}] CUDA Graph init skipped: {e}")
        _CUDA_GRAPH_ENABLED = False


def _cg_run(session, input_tensor: np.ndarray) -> np.ndarray:
    """Run inference via CUDA Graph replay if available, else fall back."""
    if getattr(session, '_cg_recorded', False):
        try:
            # update_inplace requires contiguous data — a non-contiguous
            # transpose view copies garbage to the GPU and produces
            # hallucinated faces.
            if not input_tensor.flags['C_CONTIGUOUS']:
                input_tensor = np.ascontiguousarray(input_tensor)
            session._cg_ort_inp.update_inplace(input_tensor)
            session._cg_sess.run_with_iobinding(session._cg_io)
            return session._cg_io.get_outputs()[0].numpy()
        except Exception:
            pass
    # Fallback to standard IO-bound run
    from modules.processors.frame._onnx_enhancer import run_inference
    return run_inference(session, session.get_inputs()[0].name, input_tensor)


def _has_cuda() -> bool:
    try:
        import onnxruntime
        return "CUDAExecutionProvider" in onnxruntime.get_available_providers()
    except Exception:
        return False

def pre_check() -> bool:
    model_path = os.path.join(models_dir, MODEL_FILE)
    if not os.path.exists(model_path):
        update_status(f"Downloading {MODEL_FILE}...", NAME)
        from modules.utilities import conditional_download
        conditional_download(models_dir, [MODEL_URL])
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(modules.globals.target_path):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def get_enhancer() -> Any:
    global ENHANCER, _CUDA_GRAPH_ENABLED
    with THREAD_LOCK:
        if ENHANCER is None:
            model_path = os.path.join(models_dir, MODEL_FILE)
            if not os.path.exists(model_path):
                from modules.utilities import conditional_download
                conditional_download(models_dir, [MODEL_URL])
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")
            print(f"{NAME}: Loading ONNX model from {model_path}")
            ENHANCER = create_onnx_session(model_path)
            warmup_session(ENHANCER)
            if _has_cuda():
                _init_cuda_graph(ENHANCER)
            print(f"{NAME}: Model loaded successfully.")
    return ENHANCER


def enhance_face(temp_frame: Frame, face: Face) -> Frame:
    try:
        session = get_enhancer()
    except Exception as e:
        print(f"{NAME}: {e}")
        return temp_frame
    try:
        return _enhance_face_cg(temp_frame, face, session, INPUT_SIZE)
    except Exception as e:
        print(f"{NAME}: Error during face enhancement: {e}")
        return temp_frame


def _lab_color_transfer(src: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Transfer LAB mean/std from *ref* to *src* so the two have matching
    brightness and colour statistics.  Returns a uint8 BGR image."""
    src_f = src.astype(np.float32) / 255.0
    ref_f = ref.astype(np.float32) / 255.0
    src_lab = cv2.cvtColor(src_f, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(ref_f, cv2.COLOR_BGR2LAB)
    s_mean, s_std = cv2.meanStdDev(src_lab)
    r_mean, r_std = cv2.meanStdDev(ref_lab)
    s_mean = s_mean.reshape((1, 1, 3)).astype(np.float32)
    s_std  = s_std.reshape((1, 1, 3)).astype(np.float32)
    r_mean = r_mean.reshape((1, 1, 3)).astype(np.float32)
    r_std  = r_std.reshape((1, 1, 3)).astype(np.float32)
    eps = 1e-6
    result_lab = (src_lab - s_mean) * (r_std / np.maximum(s_std, eps)) + r_mean
    result_bgr = cv2.cvtColor(result_lab, cv2.COLOR_LAB2BGR)
    return np.clip(result_bgr * 255.0, 0, 255).astype(np.uint8)


def _enhance_face_cg(
    frame: np.ndarray, face: Any, session: Any, input_size: int
) -> np.ndarray:
    """enhance_face_onnx with CUDA Graph inference path."""
    from modules.processors.frame._onnx_enhancer import (
        _get_face_affine, preprocess_face, postprocess_face, THREAD_SEMAPHORE,
    )

    M, inv_M = _get_face_affine(face, input_size)
    if M is None:
        return frame

    face_crop = cv2.warpAffine(
        frame, M, (input_size, input_size),
        flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE,
    )

    blob = preprocess_face(face_crop, input_size)
    with THREAD_SEMAPHORE:
        output = _cg_run(session, blob)
    enhanced = postprocess_face(output)

    # Colour-transfer the enhanced face to match the original crop so the
    # blend mask has no visible boundary regardless of feather radius.
    enhanced = _lab_color_transfer(enhanced, face_crop)

    h, w = frame.shape[:2]

    # Build feathered alpha mask in aligned space — same coordinate system
    # as the enhanced face, so the mask perfectly covers valid pixels.
    feather = max(0.0, min(1.0, modules.globals.edge_feather))
    alpha = np.zeros((input_size, input_size), dtype=np.float32)
    cv2.ellipse(alpha, (input_size // 2, input_size // 2),
                (int(input_size * 0.43), int(input_size * 0.43)),
                0, 0, 360, 1, -1)
    k = int(5 + feather * 86) | 1
    s = 2 + feather * 38
    alpha = cv2.GaussianBlur(alpha, (k, k), s)

    # Compute the exact output region from the inverse affine corners.
    # This matches the warped face precisely — no dependence on the
    # detection bbox, which may not align with the landmark-based affine.
    corners = np.array([[0, 0], [input_size, 0],
                        [input_size, input_size], [0, input_size]],
                       dtype=np.float32)
    transformed = (inv_M[:, :2] @ corners.T).T + inv_M[:, 2]
    x1 = int(np.floor(transformed[:, 0].min()))
    x2 = int(np.ceil(transformed[:, 0].max()))
    y1 = int(np.floor(transformed[:, 1].min()))
    y2 = int(np.ceil(transformed[:, 1].max()))

    # Small pad so the feather has room beyond the face boundary
    pad = 3
    y1p, y2p = max(0, y1 - pad), min(h, y2 + pad + 1)
    x1p, x2p = max(0, x1 - pad), min(w, x2 + pad + 1)
    crop_w, crop_h = x2p - x1p, y2p - y1p
    if crop_w <= 0 or crop_h <= 0:
        return frame

    inv_crop = inv_M.copy()
    inv_crop[0, 2] -= x1p
    inv_crop[1, 2] -= y1p

    warped_enhanced_crop = cv2.warpAffine(
        enhanced, inv_crop, (crop_w, crop_h),
        flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0),
    )
    warped_mask_crop = cv2.warpAffine(
        alpha, inv_crop, (crop_w, crop_h),
        flags=cv2.INTER_LINEAR, borderValue=0,
    )

    target_crop = frame[y1p:y2p, x1p:x2p]
    mask_3ch = warped_mask_crop[:, :, np.newaxis]
    blended = (warped_enhanced_crop.astype(np.float32) * mask_3ch +
               target_crop.astype(np.float32) * (1.0 - mask_3ch))
    frame[y1p:y2p, x1p:x2p] = np.clip(blended, 0, 255).astype(np.uint8)

    return frame


def process_frame(source_face: Face | None, temp_frame: Frame, detected_faces=None) -> Frame:
    if detected_faces:
        target_face = detected_faces[0]
    else:
        target_face = get_one_face(temp_frame)
    if target_face is None:
        return temp_frame
    return enhance_face(temp_frame, target_face)


def process_frame_v2(temp_frame: Frame) -> Frame:
    target_face = get_one_face(temp_frame)
    if target_face:
        temp_frame = enhance_face(temp_frame, target_face)
    return temp_frame


def process_frames(
    source_path: str | None, temp_frame_paths: List[str], progress: Any = None
) -> None:
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is None:
            if progress:
                progress.update(1)
            continue
        result = process_frame(None, temp_frame)
        cv2.imwrite(temp_frame_path, result)
        if progress:
            progress.update(1)


def process_image(source_path: str | None, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    if target_frame is None:
        print(f"{NAME}: Error: Failed to read target image {target_path}")
        return
    result_frame = process_frame(None, target_frame)
    cv2.imwrite(output_path, result_frame)
    print(f"{NAME}: Enhanced image saved to {output_path}")


def process_video(source_path: str | None, temp_frame_paths: List[str]) -> None:
    modules.processors.frame.core.process_video(source_path, temp_frame_paths, process_frames)
