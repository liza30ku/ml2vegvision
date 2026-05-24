"""Тонкий слой сервиса: валидация входов и делегирование в src.inference.

Все сообщения и комментарии в этом файле на русском языке.
"""
from __future__ import annotations

import logging
from typing import Any

import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from pathlib import Path
from datetime import datetime
import re

from src.inference import load_models, predict
from src.roi import RoiMode, clip_bbox
from src.settings import (
    CONFIDENCE_THRESHOLD,
    UNET_CHECKPOINT,
    VIT_CHECKPOINT,
)

logger = logging.getLogger(__name__)

ROI_MODES: tuple[RoiMode, ...] = ("auto", "prob", "mask", "bbox", "full")


def ensure_models_loaded() -> None:
    load_models()


def models_loaded() -> bool:
    from src.inference import _models_cache

    return _models_cache is not None


def parse_mode(mode: str | None) -> RoiMode:
    """Парсит режим ROI, возвращает валидный режим или кидает ValueError с русским текстом."""
    value = (mode or "auto").strip().lower()
    if value not in ROI_MODES:
        raise ValueError(f"Неверный режим: {value}. Допустимые: {', '.join(ROI_MODES)}")
    return value  # type: ignore[return-value]


def parse_bbox(bbox_str: str | None, width: int, height: int) -> tuple[int, int, int, int] | None:
    if not bbox_str or not bbox_str.strip():
        return None
    parts = [p.strip() for p in bbox_str.replace(";", ",").split(",")]
    if len(parts) != 4:
        raise ValueError("bbox должен содержать 4 целых числа: x1,y1,x2,y2")
    try:
        x1, y1, x2, y2 = (int(p) for p in parts)
    except ValueError as exc:
        raise ValueError("bbox должен содержать только целые числа") from exc
    return clip_bbox((x1, y1, x2, y2), height, width)


def read_mask_array(mask_bytes: bytes, target_width: int, target_height: int) -> np.ndarray:
    image = Image.open(BytesIO(mask_bytes))
    mask = np.array(image.convert("L"))
    if mask.shape[0] != target_height or mask.shape[1] != target_width:
        mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)
    return mask


def to_api_response(raw: dict[str, Any]) -> dict[str, Any]:
    """Elimina arrays numpy y campos internos no expuestos por API."""
    # Преобразуем топ-3 классов в сериализуемый формат
    top3 = [
        {"class": item["class"], "confidence": item["confidence"]}
        for item in raw.get("top3", [])
    ]
    # Процент доверия, округлённый до целых для удобства пользователя
    confidence_percent = int(round(float(raw.get("confidence", 0.0)) * 100))
    return {
        "disease": raw["disease"],
        "confidence": float(raw["confidence"]),
        "confidence_percent": confidence_percent,
        "top3": top3,
        "mode_requested": raw["mode_requested"],
        "mode_used": raw["mode_used"],
        "fallback_used": raw["fallback_used"],
        "bbox_used": raw.get("bbox_used"),
        "mask_used": raw["mask_used"],
        "prob_used": raw["prob_used"],
        "needs_agronomist": raw["needs_agronomist"],
        "message": raw["message"],
        "image_size": raw["image_size"],
        "segmentation": raw.get("segmentation", {}),
    }


def _safe_filename(name: str) -> str:
    """Генерирует безопасное имя файла на основе оригинального имени и текущего времени."""
    base = Path(name).stem if name else "upload"
    # Оставляем только буквенно-цифровые, дефисы и подчёркивания
    base = re.sub(r"[^A-Za-z0-9_-]", "_", base)
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    return f"{base}_{ts}.png"


def run_predict(
    image_bytes: bytes,
    *,
    mode: str | None = "auto",
    original_filename: str | None = None,
) -> dict[str, Any]:
    """Запускает единый pipeline предсказания и создаёт визуализацию.

    Внешние пользователи передают только `image`. Параметры `mask`/`bbox` больше
    не принимаются в основном сценарии (убраны из публичного endpoint).
    """
    ensure_models_loaded()
    roi_mode = parse_mode(mode)

    from src.inference import read_image_rgb, visualize_prediction, predict

    image_rgb = read_image_rgb(image_bytes)
    h, w = image_rgb.shape[:2]

    logger.info("predict mode=%s image=%dx%d", roi_mode, w, h)

    # Вызов централизованного predict: он сам запустит UNet при необходимости
    raw = predict(
        image_bytes,
        mode=roi_mode,
        run_seg_if_needed=True,
    )

    # Создаём директорию для визуализаций
    out_dir = Path("outputs") / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_name = _safe_filename(original_filename or "upload")
    vis_path = out_dir / vis_name

    try:
        visualize_prediction(image_rgb, raw, save_path=str(vis_path))
        raw["visualization_path"] = str(vis_path)
        raw["visualization_created"] = True
    except Exception:
        logger.exception("Не удалось создать визуализацию")
        raw["visualization_path"] = None
        raw["visualization_created"] = False

    api_resp = to_api_response(raw)
    # Добавляем поля визуализации в ответ API
    api_resp["visualization_path"] = raw.get("visualization_path")
    api_resp["visualization_created"] = bool(raw.get("visualization_created"))
    return api_resp


def get_model_info() -> dict[str, Any]:
    bundle = load_models()
    return {
        "segmentation_model": UNET_CHECKPOINT.name,
        "classifier_model": VIT_CHECKPOINT.name,
        "num_classes": len(bundle["idx_to_class"]),
        "classes": bundle["idx_to_class"],
        "device": str(bundle["device"]),
        "vit_best_val_acc": bundle["cls_meta"].get("best_val_acc"),
        "confidence_threshold": CONFIDENCE_THRESHOLD,
        "roi_modes": list(ROI_MODES),
    }
