from __future__ import annotations

from typing import Literal

import numpy as np

from src.settings import BBOX_PADDING_RATIO, MIN_MASK_AREA_RATIO, SEG_THRESHOLD

RoiMode = Literal["auto", "prob", "mask", "bbox", "full"]


def _as_float_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.ndim == 3:
        arr = arr[..., 0]
    arr = arr.astype(np.float32)
    if arr.max() > 1.0:
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def prob_to_mask(prob_map: np.ndarray, threshold: float = SEG_THRESHOLD) -> np.ndarray:
    prob = np.asarray(prob_map, dtype=np.float32)
    if prob.ndim == 3:
        prob = prob.squeeze()
    if prob.max() > 1.0:
        prob = prob / 255.0
    return (prob >= threshold).astype(np.uint8)


def mask_area_ratio(mask: np.ndarray, height: int, width: int) -> float:
    binary = _as_float_mask(mask) > 0.5
    return float(binary.sum()) / max(height * width, 1)


def mask_to_bbox(
    mask: np.ndarray,
    height: int,
    width: int,
    padding_ratio: float = BBOX_PADDING_RATIO,
) -> tuple[int, int, int, int] | None:
    binary = (_as_float_mask(mask) > 0.5).astype(np.uint8)
    ys, xs = np.where(binary > 0)
    if len(xs) == 0:
        return None
    x1, x2 = int(xs.min()), int(xs.max())
    y1, y2 = int(ys.min()), int(ys.max())
    pad_x = int((x2 - x1 + 1) * padding_ratio)
    pad_y = int((y2 - y1 + 1) * padding_ratio)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(width - 1, x2 + pad_x)
    y2 = min(height - 1, y2 + pad_y)
    return x1, y1, x2, y2


def clip_bbox(bbox: tuple[int, int, int, int], height: int, width: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = int(np.clip(x1, 0, width - 1))
    x2 = int(np.clip(x2, 0, width - 1))
    y1 = int(np.clip(y1, 0, height - 1))
    y2 = int(np.clip(y2, 0, height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def is_valid_mask(mask: np.ndarray | None, height: int, width: int) -> bool:
    if mask is None:
        return False
    return mask_area_ratio(mask, height, width) >= MIN_MASK_AREA_RATIO


def is_valid_bbox(bbox: tuple[int, int, int, int] | None) -> bool:
    if bbox is None:
        return False
    x1, y1, x2, y2 = bbox
    return (x2 - x1) >= 4 and (y2 - y1) >= 4


def crop_image(image_rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    return image_rgb[y1 : y2 + 1, x1 : x2 + 1].copy()


def apply_mask_crop(image_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    mask_resized = _as_float_mask(mask)
    if mask_resized.shape[:2] != (h, w):
        import cv2

        mask_resized = cv2.resize(mask_resized, (w, h), interpolation=cv2.INTER_NEAREST)
    binary = mask_resized > 0.5
    if not binary.any():
        return image_rgb.copy()
    out = image_rgb.copy()
    out[~binary] = 0
    bbox = mask_to_bbox(binary.astype(np.uint8), h, w)
    if bbox is None:
        return out
    return crop_image(out, bbox)


def prepare_roi(
    image_rgb: np.ndarray,
    *,
    mode: RoiMode = "auto",
    prob_map: np.ndarray | None = None,
    mask: np.ndarray | None = None,
    bbox: tuple[int, int, int, int] | None = None,
) -> dict:
  
    h, w = image_rgb.shape[:2]
    requested = mode
    fallback_used = False
    mode_used: RoiMode = "full"

    prob_used = False
    mask_used = False
    bbox_used: tuple[int, int, int, int] | None = None

    def _try_prob() -> tuple[np.ndarray, RoiMode] | None:
        nonlocal prob_used, mask_used, bbox_used
        if prob_map is None:
            return None
        prob_used = True
        derived_mask = prob_to_mask(prob_map)
        if not is_valid_mask(derived_mask, h, w):
            return None
        mask_used = True
        bbox_used = mask_to_bbox(derived_mask, h, w)
        return apply_mask_crop(image_rgb, derived_mask), "prob"

    def _try_mask() -> tuple[np.ndarray, RoiMode] | None:
        nonlocal mask_used, bbox_used
        if mask is None or not is_valid_mask(mask, h, w):
            return None
        mask_used = True
        bbox_used = mask_to_bbox(mask, h, w)
        return apply_mask_crop(image_rgb, mask), "mask"

    def _try_bbox() -> tuple[np.ndarray, RoiMode] | None:
        nonlocal bbox_used
        if bbox is None:
            return None
        clipped = clip_bbox(bbox, h, w)
        if not is_valid_bbox(clipped):
            return None
        bbox_used = clipped
        return crop_image(image_rgb, clipped), "bbox"

    candidates: list[RoiMode]
    if mode == "auto":
        candidates = ["prob", "mask", "bbox", "full"]
    else:
        candidates = [mode]

    roi_image = image_rgb.copy()
    for candidate in candidates:
        if candidate == "prob":
            result = _try_prob()
            if result is not None:
                roi_image, mode_used = result
                break
        elif candidate == "mask":
            result = _try_mask()
            if result is not None:
                roi_image, mode_used = result
                break
        elif candidate == "bbox":
            result = _try_bbox()
            if result is not None:
                roi_image, mode_used = result
                break
        elif candidate == "full":
            mode_used = "full"
            roi_image = image_rgb.copy()
            break

    if mode != "auto" and mode != mode_used and mode != "full":
        fallback_used = True
        if mode_used == "full":
            fallback_used = True

    if mode != "auto" and candidates[0] != "full" and mode_used == "full":
        fallback_used = True

    return {
        "roi_image": roi_image,
        "mode_requested": requested,
        "mode_used": mode_used,
        "fallback_used": fallback_used,
        "prob_used": prob_used,
        "mask_used": mask_used,
        "bbox_used": list(bbox_used) if bbox_used else None,
    }
