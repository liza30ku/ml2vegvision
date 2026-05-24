from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import albumentations as A
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from albumentations.pytorch import ToTensorV2
from PIL import Image

from src.models import (
    CheckpointUNet,
    ViTClassifier,
    load_classifier_model,
    load_segmentation_model,
    resolve_device,
)
from src.roi import (
    RoiMode,
    mask_to_bbox,
    prepare_roi,
    prob_to_mask,
)
from src.settings import (
    CONFIDENCE_THRESHOLD,
    IMAGENET_MEAN,
    IMAGENET_STD,
    SEG_THRESHOLD,
    UNET_IMAGE_SIZE,
    VIT_IMAGE_SIZE,
    VIT_RESIZE,
)

_models_cache: dict[str, Any] | None = None


def _seg_transform() -> A.Compose:
    return A.Compose(
        [
            A.Resize(UNET_IMAGE_SIZE, UNET_IMAGE_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def _cls_transform() -> A.Compose:
    return A.Compose(
        [
            A.Resize(VIT_RESIZE, VIT_RESIZE),
            A.CenterCrop(VIT_IMAGE_SIZE, VIT_IMAGE_SIZE),
            A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ToTensorV2(),
        ]
    )


def load_models(force_reload: bool = False) -> dict[str, Any]:
    global _models_cache
    if _models_cache is not None and not force_reload:
        return _models_cache

    device = resolve_device()
    seg_model, _ = load_segmentation_model(device=device)
    cls_model, idx_to_class, class_to_idx, cls_meta, _ = load_classifier_model(device=device)
    _models_cache = {
        "device": device,
        "seg_model": seg_model,
        "cls_model": cls_model,
        "idx_to_class": idx_to_class,
        "class_to_idx": class_to_idx,
        "cls_meta": cls_meta,
        "seg_transform": _seg_transform(),
        "cls_transform": _cls_transform(),
    }
    return _models_cache


def read_image_rgb(source: str | Path | bytes) -> np.ndarray:
    if isinstance(source, (str, Path)):
        bgr = cv2.imread(str(source))
        if bgr is None:
            raise ValueError(f"No se pudo leer la imagen: {source}")
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image = Image.open(BytesIO(source)).convert("RGB")
    return np.array(image)


def preprocess_image(image_rgb: np.ndarray, *, for_segmentation: bool = True) -> torch.Tensor:
    bundle = load_models()
    transform = bundle["seg_transform"] if for_segmentation else bundle["cls_transform"]
    tensor = transform(image=image_rgb)["image"].unsqueeze(0)
    return tensor.to(bundle["device"])


@torch.inference_mode()
def run_segmentation(
    image_rgb: np.ndarray,
    seg_model: CheckpointUNet | None = None,
    device: torch.device | None = None,
) -> dict[str, Any]:
    bundle = load_models()
    seg_model = seg_model or bundle["seg_model"]
    device = device or bundle["device"]

    h, w = image_rgb.shape[:2]
    tensor = bundle["seg_transform"](image=image_rgb)["image"].unsqueeze(0).to(device)
    logits = seg_model(tensor)
    prob_small = torch.sigmoid(logits)[0, 0].cpu().numpy().astype(np.float32)
    prob_map = cv2.resize(prob_small, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = prob_to_mask(prob_map, threshold=SEG_THRESHOLD)
    bbox = mask_to_bbox(mask, h, w)

    return {
        "prob_map": prob_map,
        "mask": mask,
        "bbox": bbox,
        "logits": logits,
    }


@torch.inference_mode()
def run_classification(
    roi_rgb: np.ndarray,
    cls_model: ViTClassifier | None = None,
    idx_to_class: dict[int, str] | None = None,
    device: torch.device | None = None,
    top_k: int = 3,
) -> dict[str, Any]:
    bundle = load_models()
    cls_model = cls_model or bundle["cls_model"]
    idx_to_class = idx_to_class or bundle["idx_to_class"]
    device = device or bundle["device"]

    tensor = bundle["cls_transform"](image=roi_rgb)["image"].unsqueeze(0).to(device)
    logits = cls_model(tensor)
    probs = F.softmax(logits, dim=1)[0].cpu().numpy()
    order = np.argsort(probs)[::-1][:top_k]

    top3 = [
        {
            "class": idx_to_class[int(i)],
            "confidence": float(probs[int(i)]),
        }
        for i in order
    ]
    best_idx = int(order[0])
    confidence = float(probs[best_idx])
    disease = idx_to_class[best_idx]

    return {
        "disease": disease,
        "confidence": confidence,
        "top3": top3,
        "probs": probs,
        "class_index": best_idx,
    }


def predict(
    image: str | Path | bytes,
    *,
    mask: np.ndarray | None = None,
    bbox: tuple[int, int, int, int] | None = None,
    prob_map: np.ndarray | None = None,
    mode: RoiMode = "auto",
    run_seg_if_needed: bool = True,
) -> dict[str, Any]:
    image_rgb = read_image_rgb(image)
    h, w = image_rgb.shape[:2]

    seg_outputs: dict[str, Any] = {}
    if run_seg_if_needed and mode in ("auto", "prob", "mask", "bbox") and prob_map is None and mask is None and bbox is None:
        seg_outputs = run_segmentation(image_rgb)
        prob_map = seg_outputs.get("prob_map")
        if mask is None:
            mask = seg_outputs.get("mask")
        if bbox is None:
            bbox = seg_outputs.get("bbox")

    roi_info = prepare_roi(
        image_rgb,
        mode=mode,
        prob_map=prob_map,
        mask=mask,
        bbox=bbox,
    )
    cls_result = run_classification(roi_info["roi_image"])

    confidence = cls_result["confidence"]
    needs_agronomist = confidence < CONFIDENCE_THRESHOLD
    message = (
        "Низкая уверенность: рекомендуется проверка агронома."
        if needs_agronomist
        else "Автоматическая предсказание в пределах порога доверия."
    )

    return {
        **cls_result,
        "mode_requested": roi_info["mode_requested"],
        "mode_used": roi_info["mode_used"],
        "fallback_used": roi_info["fallback_used"],
        "bbox_used": roi_info["bbox_used"],
        "mask_used": roi_info["mask_used"],
        "prob_used": roi_info["prob_used"],
        "needs_agronomist": needs_agronomist,
        "message": message,
        "image_size": [w, h],
        "segmentation": {
            "has_prob_map": prob_map is not None,
            "has_mask": mask is not None,
            "has_bbox": bbox is not None,
        },
        "seg_prob_map": prob_map,
        "seg_mask": mask,
    }


def visualize_prediction(
    image_rgb: np.ndarray,
    result: dict[str, Any],
    *,
    save_path: str | Path | None = None,
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = result.get("seg_mask")
    prob_map = result.get("seg_prob_map")
    panels = [("Input", image_rgb)]
    if prob_map is not None:
        panels.append(("Prob map", prob_map))
    if mask is not None:
        panels.append(("Mask", mask))

    overlay = image_rgb.copy()
    if mask is not None and mask.shape[:2] == overlay.shape[:2]:
        overlay = overlay.copy()
        overlay[mask > 0, 0] = 255
        overlay[mask > 0, 1] = (overlay[mask > 0, 1] * 0.3).astype(np.uint8)
    panels.append(("Overlay", overlay))

    fig, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    if len(panels) == 1:
        axes = [axes]
    conf_pct = int(round(float(result.get("confidence", 0.0)) * 100))
    title = (
        f"{result['disease']} ({conf_pct}%)\n"
        f"mode={result['mode_used']} fallback={result['fallback_used']}"
    )
    fig.suptitle(title, fontsize=10)
    for ax, (name, img) in zip(axes, panels):
        if img.ndim == 2:
            ax.imshow(img, cmap="gray", vmin=0, vmax=1 if img.max() <= 1 else 255)
        else:
            ax.imshow(img)
        ax.set_title(name)
        ax.axis("off")
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    fig.canvas.draw()
    canvas = np.asarray(fig.canvas.buffer_rgba())
    plt.close(fig)
    vis = canvas[..., :3].copy()
    return vis
