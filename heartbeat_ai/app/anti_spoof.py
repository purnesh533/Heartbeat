"""
Silent-Face Anti-Spoofing filter.

Uses MiniFASNetV2 — a ~435 KB depthwise-separable CNN from Minivision trained
to distinguish real faces from spoofs (printed photos, tablet/phone screens,
video replays, 3-D masks).

Source: https://github.com/minivision-ai/Silent-Face-Anti-Spoofing

The model weight file is auto-downloaded into ``heartbeat_ai/models/`` on
first use.  PyTorch is required (already installed as a project dependency).

Label convention (matches training data in the original repo):
  class 0 → spoof  (photo / screen / mask)
  class 1 → real   (live person)

``liveness_score = softmax(model_output)[0, 1]``
Faces with score < ``threshold`` are removed from FaceDetectionResult.

Face crop: the bounding box is expanded by a scale factor (2.7×) before
being resized to 80 × 80, giving the model enough context to distinguish
skin texture from flat printed/digital replicas.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("heartbeat")

# ── constants ──────────────────────────────────────────────────────────
_MODEL_URL = (
    "https://raw.githubusercontent.com/minivision-ai/"
    "Silent-Face-Anti-Spoofing/master/resources/anti_spoof_models/"
    "2.7_80x80_MiniFASNetV2.pth"
)
_MODEL_FILENAME = "2.7_80x80_MiniFASNetV2.pth"

_FACE_SCALE = 2.7   # expand face box before cropping (encodes in filename)
_INPUT_SIZE = 80    # model expects 80 × 80 RGB input

_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── PyTorch model architecture (MiniFASNetV2) ──────────────────────────
# Defined at module level inside a try/except so the rest of the file
# imports cleanly even when torch is absent.
try:
    import torch
    import torch.nn as nn
    from torch.nn import (
        BatchNorm1d, BatchNorm2d, Conv2d, Linear, PReLU,
    )
    _TORCH_OK = True

    # ------------------------------------------------------------------
    # Building blocks — attribute names must match the original repo
    # exactly so that the pre-trained .pth state_dict loads without error.
    # ------------------------------------------------------------------

    class _CB(nn.Module):
        """Conv2d + BatchNorm2d + PReLU  (Conv_block in original)."""
        def __init__(self, in_c: int, out_c: int,
                     kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
            super().__init__()
            self.conv  = Conv2d(in_c, out_c, kernel_size=kernel,
                                stride=stride, padding=padding,
                                groups=groups, bias=False)
            self.bn    = BatchNorm2d(out_c)
            self.prelu = PReLU(out_c)

        def forward(self, x):
            return self.prelu(self.bn(self.conv(x)))

    class _LB(nn.Module):
        """Conv2d + BatchNorm2d, no activation  (Linear_block in original)."""
        def __init__(self, in_c: int, out_c: int,
                     kernel=(1, 1), stride=(1, 1), padding=(0, 0), groups=1):
            super().__init__()
            self.conv = Conv2d(in_c, out_c, kernel_size=kernel,
                               stride=stride, padding=padding,
                               groups=groups, bias=False)
            self.bn   = BatchNorm2d(out_c)

        def forward(self, x):
            return self.bn(self.conv(x))

    class _DW(nn.Module):
        """Depth-wise separable block  (Depth_Wise in original)."""
        def __init__(self, in_c: int, out_c: int, residual=False,
                     kernel=(3, 3), stride=(2, 2), padding=(1, 1), groups=1):
            super().__init__()
            self.conv     = _CB(in_c, groups, kernel=(1, 1))
            self.conv_dw  = _CB(groups, groups, kernel=kernel,
                                stride=stride, padding=padding, groups=groups)
            self.project  = _LB(groups, out_c, kernel=(1, 1))
            self.residual = residual

        def forward(self, x):
            shortcut = x
            x = self.project(self.conv_dw(self.conv(x)))
            return (shortcut + x) if self.residual else x

    class _Res(nn.Module):
        """Stack of residual Depth-Wise blocks.

        The original model was trained with neural architecture search, so
        each block in the stack may have a *different* intermediate group
        count.  ``groups_per_block`` is a list with one entry per block.
        """
        def __init__(self, c: int, groups_per_block: List[int],
                     kernel=(3, 3), stride=(1, 1), padding=(1, 1)):
            super().__init__()
            self.model = nn.Sequential(*[
                _DW(c, c, residual=True,
                    kernel=kernel, stride=stride, padding=padding, groups=g)
                for g in groups_per_block
            ])

        def forward(self, x):
            return self.model(x)

    class _Flatten(nn.Module):
        def forward(self, x):
            return x.view(x.size(0), -1)

    class _MiniFASNetV2(nn.Module):
        """
        MiniFASNetV2 with architecture inferred from checkpoint shapes.

        The pre-trained weights were produced by a channel-pruning / NAS
        procedure, so each layer has a unique channel width that cannot be
        predicted upfront.  Pass ``arch`` — a dict produced by
        ``_infer_arch()`` — to build a model that exactly matches the
        downloaded checkpoint.

        Input : float32 tensor  [B, 3, 80, 80]  (ImageNet normalised)
        Output: logits          [B, num_classes]
        """
        def __init__(self, arch: dict):
            super().__init__()
            c1    = arch["c1"]
            g23   = arch["g23"];   out_23  = arch["out_23"]
            g3    = arch["g3"]
            g34   = arch["g34"];   out_34  = arch["out_34"]
            g4    = arch["g4"]
            g45   = arch["g45"];   out_45  = arch["out_45"]
            g5    = arch["g5"]
            c6sep = arch["c6sep"]; k6      = arch["k6"]
            emb   = arch["emb"];   nc      = arch["nc"]

            self.conv1          = _CB(3, c1, kernel=(3, 3),
                                      stride=(2, 2), padding=(1, 1))
            self.conv2_dw       = _CB(c1, c1, kernel=(3, 3),
                                      stride=(1, 1), padding=(1, 1), groups=c1)
            self.conv_23        = _DW(c1, out_23, kernel=(3, 3),
                                      stride=(2, 2), padding=(1, 1), groups=g23)
            self.conv_3         = _Res(out_23,  g3)
            self.conv_34        = _DW(out_23, out_34, kernel=(3, 3),
                                      stride=(2, 2), padding=(1, 1), groups=g34)
            self.conv_4         = _Res(out_34, g4)
            self.conv_45        = _DW(out_34, out_45, kernel=(3, 3),
                                      stride=(2, 2), padding=(1, 1), groups=g45)
            self.conv_5         = _Res(out_45, g5)
            self.conv_6_sep     = _CB(out_45, c6sep, kernel=(1, 1))
            self.conv_6_dw      = _LB(c6sep, c6sep, kernel=k6, groups=c6sep)
            self.conv_6_flatten = _Flatten()
            self.linear = Linear(c6sep, emb, bias=False)
            self.bn     = BatchNorm1d(emb)
            self.drop   = nn.Dropout(p=0.75)
            self.prob   = Linear(emb, nc, bias=("prob.bias" in arch))

        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2_dw(x)
            x = self.conv_23(x)
            x = self.conv_3(x)
            x = self.conv_34(x)
            x = self.conv_4(x)
            x = self.conv_45(x)
            x = self.conv_5(x)
            x = self.conv_6_sep(x)
            x = self.conv_6_dw(x)
            x = self.conv_6_flatten(x)
            x = self.linear(x)
            x = self.bn(x)
            x = self.drop(x)
            return self.prob(x)

    def _infer_arch(state: dict) -> dict:
        """Read tensor shapes from a state_dict to reconstruct the arch dict."""

        def n(key: str) -> int:
            return state[key].shape[0]

        # conv_3 / conv_4 / conv_5 each have variable groups per block
        def block_groups(prefix: str) -> List[int]:
            gs, i = [], 0
            while f"{prefix}.model.{i}.conv.conv.weight" in state:
                gs.append(state[f"{prefix}.model.{i}.conv.conv.weight"].shape[0])
                i += 1
            return gs

        k6h = state["conv_6_dw.conv.weight"].shape[2]
        arch = dict(
            c1      = n("conv1.conv.weight"),
            g23     = n("conv_23.conv.conv.weight"),
            out_23  = n("conv_23.project.conv.weight"),
            g3      = block_groups("conv_3"),
            g34     = n("conv_34.conv.conv.weight"),
            out_34  = n("conv_34.project.conv.weight"),
            g4      = block_groups("conv_4"),
            g45     = n("conv_45.conv.conv.weight"),
            out_45  = n("conv_45.project.conv.weight"),
            g5      = block_groups("conv_5"),
            c6sep   = n("conv_6_sep.conv.weight"),
            k6      = (k6h, k6h),
            emb     = state["linear.weight"].shape[0],
            nc      = n("prob.weight"),
        )
        if "prob.bias" in state:
            arch["prob.bias"] = True
        return arch

except ImportError:
    _TORCH_OK = False


# ── Public filter class ────────────────────────────────────────────────

class AntiSpoofFilter:
    """
    Secondary liveness check applied to every detected face box.

    Faces whose liveness score (probability of being a real person) falls
    below ``threshold`` are classified as spoofs and removed from the result.

    The model is loaded lazily on the first call to ``filter()``.  If PyTorch
    is unavailable or the model fails to load, the filter degrades gracefully:
    all faces are passed through unchanged.

    Parameters
    ----------
    models_dir:
        Directory where the model weight file is cached.
        File is auto-downloaded if absent.
    threshold:
        Liveness score below which a face is treated as spoof.
        Default 0.55 — conservative; lower → fewer rejections.
    """

    def __init__(self, models_dir: Path, threshold: float = 0.55) -> None:
        self._models_dir  = models_dir
        self._threshold   = threshold
        self._model       = None           # loaded on first use
        self._real_idx    = 2              # class index for "real face" (updated at load)
        self._log_first   = True           # emit one verbose diagnostic on first inference
        self._degraded    = False
        self._degraded_reason: str = ""
        self.available    = False

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    @property
    def model_name(self) -> str:
        if self._degraded:
            return f"anti_spoof:disabled({self._degraded_reason})"
        if self._model is not None:
            return "MiniFASNetV2"
        return "anti_spoof:not_loaded"

    def ensure_loaded(self) -> bool:
        """Load (and possibly download) the model.  Returns True on success."""
        if self._model is not None:
            return True
        if self._degraded:
            return False
        if not _TORCH_OK:
            self._fail("PyTorch not installed")
            return False
        try:
            path  = self._ensure_model_file()
            ckpt  = torch.load(path, map_location="cpu", weights_only=True)
            state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
            # Models saved with DataParallel prefix every key with "module."
            if all(k.startswith("module.") for k in state):
                state = {k[len("module."):]: v for k, v in state.items()}
            # Build model whose channel widths match this checkpoint exactly
            arch  = _infer_arch(state)
            model = _MiniFASNetV2(arch)
            model.load_state_dict(state, strict=True)
            model.eval()
            self._model    = model
            self.available = True
            # Training convention inferred from checkpoint output shape:
            #   2-class: 0=spoof,       1=real
            #   3-class: 0=spoof-print, 1=spoof-video, 2=real
            self._real_idx = 1 if arch["nc"] == 2 else 2
            self._log_first = True   # emit one verbose diagnostic on first face
            logger.info(
                "AntiSpoofFilter: MiniFASNetV2 loaded  classes=%d  real_idx=%d  threshold=%.2f",
                arch["nc"], self._real_idx, self._threshold,
            )
            return True
        except Exception as exc:
            self._fail(str(exc))
            return False

    def filter(
        self,
        frame_bgr: np.ndarray,
        boxes_xywh: List[Tuple[int, int, int, int]],
        confidences: List[float],
    ) -> Tuple[List[Tuple[int, int, int, int]], List[float], int]:
        """
        Return ``(filtered_boxes, filtered_confs, n_spoof_removed)``.

        Spoof faces are dropped; live faces are kept.  All faces pass through
        unchanged if the model is unavailable.
        """
        if not boxes_xywh:
            return [], [], 0

        if self._model is None:
            self.ensure_loaded()
        if self._model is None:
            return boxes_xywh, confidences, 0   # graceful fallback

        fh, fw = frame_bgr.shape[:2]
        kept_boxes: List[Tuple[int, int, int, int]] = []
        kept_confs: List[float] = []
        n_removed = 0

        for i, (x, y, bw, bh) in enumerate(boxes_xywh):
            crop = _make_crop(frame_bgr, x, y, bw, bh, _FACE_SCALE,
                              _INPUT_SIZE, fw, fh)
            if crop is None:
                # Can't crop → keep to avoid false negatives
                kept_boxes.append((x, y, bw, bh))
                kept_confs.append(confidences[i] if i < len(confidences) else 0.0)
                continue

            score = self._liveness_score(crop)
            if score >= self._threshold:
                kept_boxes.append((x, y, bw, bh))
                kept_confs.append(confidences[i] if i < len(confidences) else 0.0)
                logger.debug("AntiSpoof: REAL  score=%.3f  box=(%d,%d,%d,%d)",
                             score, x, y, bw, bh)
            else:
                n_removed += 1
                logger.info("AntiSpoof: SPOOF score=%.3f  box=(%d,%d,%d,%d) — suppressed",
                            score, x, y, bw, bh)

        return kept_boxes, kept_confs, n_removed

    def liveness_scores(
        self,
        frame_bgr: np.ndarray,
        boxes_xywh: List[Tuple[int, int, int, int]],
    ) -> List[float]:
        """Return raw liveness scores (0–1) for each box, for HUD display."""
        if self._model is None:
            return [1.0] * len(boxes_xywh)
        fh, fw = frame_bgr.shape[:2]
        scores = []
        for (x, y, bw, bh) in boxes_xywh:
            crop = _make_crop(frame_bgr, x, y, bw, bh, _FACE_SCALE,
                              _INPUT_SIZE, fw, fh)
            scores.append(self._liveness_score(crop) if crop is not None else 1.0)
        return scores

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _liveness_score(self, crop_bgr: np.ndarray) -> float:
        """Run one forward pass; return real-face probability (0–1)."""
        tensor = _preprocess(crop_bgr)
        with torch.no_grad():
            logits = self._model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
        if self._log_first:
            self._log_first = False
            logger.info(
                "AntiSpoof class probs (first face): %s  → using class[%d]=%.3f as liveness",
                [f"{p:.3f}" for p in probs.tolist()],
                self._real_idx,
                float(probs[self._real_idx]),
            )
        return float(probs[self._real_idx])

    def _ensure_model_file(self) -> Path:
        """Return local path to model weights, downloading if necessary."""
        self._models_dir.mkdir(parents=True, exist_ok=True)
        dest = self._models_dir / _MODEL_FILENAME
        if not dest.exists():
            logger.info("AntiSpoofFilter: downloading %s ...", _MODEL_FILENAME)
            urllib.request.urlretrieve(_MODEL_URL, dest)
            logger.info("AntiSpoofFilter: download complete → %s", dest)
        return dest

    def _fail(self, reason: str) -> None:
        self._degraded        = True
        self._degraded_reason = reason
        self.available        = False
        logger.warning("AntiSpoofFilter disabled: %s", reason)


# ── Module helpers ─────────────────────────────────────────────────────

def _make_crop(
    frame: np.ndarray,
    x: int, y: int, bw: int, bh: int,
    scale: float, size: int,
    fw: int, fh: int,
) -> Optional[np.ndarray]:
    """Expand the face box by *scale*, crop from frame, resize to size×size."""
    cx = x + bw / 2.0
    cy = y + bh / 2.0
    half_w = bw * scale / 2.0
    half_h = bh * scale / 2.0
    x1 = max(0, int(cx - half_w))
    y1 = max(0, int(cy - half_h))
    x2 = min(fw, int(cx + half_w))
    y2 = min(fh, int(cy + half_h))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None
    crop = frame[y1:y2, x1:x2]
    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_LINEAR)


def _preprocess(crop_bgr: np.ndarray):
    """BGR uint8 80×80 → normalised float32 tensor [1, 3, 80, 80]."""
    rgb  = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    img  = rgb.astype(np.float32) / 255.0
    img  = (img - _MEAN) / _STD
    # HWC → CHW → add batch dim
    arr  = img.transpose(2, 0, 1)
    return torch.from_numpy(arr).unsqueeze(0)
