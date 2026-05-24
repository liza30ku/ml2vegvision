from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from src.inference import load_models, predict, visualize_prediction
from src.settings import EVAL_DIR, TEST_IMAGES_DIR, VIS_DIR

_FILENAME_HINTS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"anthracnose", re.I), "Cucumber_Anthracnose"),
    (re.compile(r"bacterial\s*wilt", re.I), "Cucumber_Bacterial Wilt"),
    (re.compile(r"fresh\s*leaf", re.I), "Cucumber_Fresh Leaf"),
    (re.compile(r"TgS", re.I), "Tomato___Target_Spot"),
]


def infer_label_from_filename(filename: str, class_names: set[str]) -> str | None:
    for pattern, label in _FILENAME_HINTS:
        if pattern.search(filename) and label in class_names:
            return label
    return None


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    raise TypeError(f"No serializable: {type(obj)}")


def _serialize_prediction(result: dict[str, Any]) -> dict[str, Any]:
    out = {k: v for k, v in result.items() if k not in ("seg_prob_map", "seg_mask")}
    out["segmentation"] = result.get("segmentation", {})
    return out


def run_test_images_evaluation(
    images_dir: Path | None = None,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    images_dir = images_dir or TEST_IMAGES_DIR
    output_dir = output_dir or EVAL_DIR
    vis_dir = VIS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_models()
    class_names = set(bundle["class_to_idx"].keys())
    idx_to_class = bundle["idx_to_class"]

    image_paths = sorted(
        p
        for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not image_paths:
        raise FileNotFoundError(f"No hay imágenes en {images_dir}")

    per_image: list[dict[str, Any]] = []
    y_true: list[str] = []
    y_pred: list[str] = []

    mode_counts: dict[str, int] = {}
    fallback_count = 0

    for path in image_paths:
        result = predict(path, mode="auto")
        hint = infer_label_from_filename(path.name, class_names)
        serialized = _serialize_prediction(result)
        record = {
            "image": path.name,
            "hint_label": hint,
            **serialized,
        }
        per_image.append(record)
        mode_counts[result["mode_used"]] = mode_counts.get(result["mode_used"], 0) + 1
        if result["fallback_used"]:
            fallback_count += 1

        if hint:
            y_true.append(hint)
            y_pred.append(result["disease"])

        image_rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
        visualize_prediction(
            image_rgb,
            result,
            save_path=vis_dir / f"{path.stem}_eval.png",
        )

    metrics: dict[str, Any] = {
        "segmentation_gt_available": False,
        "note": (
            "Нет масок GT в test_images. "
            "Метрики сегментации: только визуальный анализ в outputs/visualizations/."
        ),
        "images_evaluated": len(image_paths),
        "mode_distribution": mode_counts,
        "fallback_count": fallback_count,
    }

    if y_true:
        labels_sorted = sorted(class_names.intersection(set(y_true) | set(y_pred)))
        cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
        metrics["classification"] = {
            "labeled_images": len(y_true),
            "label_source": "filename_hints",
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_precision": float(
                precision_score(y_true, y_pred, labels=labels_sorted, average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(y_true, y_pred, labels=labels_sorted, average="macro", zero_division=0)
            ),
            "macro_f1": float(
                f1_score(y_true, y_pred, labels=labels_sorted, average="macro", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(y_true, y_pred, labels=labels_sorted, average="weighted", zero_division=0)
            ),
            "confusion_matrix": {
                "labels": labels_sorted,
                "matrix": cm.tolist(),
            },
            "classification_report": classification_report(
                y_true, y_pred, labels=labels_sorted, zero_division=0
            ),
        }
    else:
        metrics["classification"] = {
            "labeled_images": 0,
            "note": "Нет меток, выводимых из имени файла.",
        }

    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_info": {
            "classes": idx_to_class,
            "vit_best_val_acc": bundle["cls_meta"].get("best_val_acc"),
        },
        "metrics": metrics,
        "per_image": per_image,
    }

    report_path = output_dir / "evaluation_report.json"
    summary_path = output_dir / "summary.txt"

    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=_json_default)

    _write_summary(summary_path, report)
    return report


def _write_summary(path: Path, report: dict[str, Any]) -> None:
    lines = [
        "VegVision — Сведения оценки",
        "=" * 40,
        f"Imágenes: {report['metrics']['images_evaluated']}",
        f"Modos usados: {report['metrics']['mode_distribution']}",
        f"Fallbacks: {report['metrics']['fallback_count']}",
        "",
        report["metrics"].get("note", ""),
    ]
    cls = report["metrics"].get("classification", {})
    if cls.get("labeled_images", 0) > 0:
        lines.extend(
            [
                "",
                "Классификация (классы по названию файла):",
                f"  accuracy: {cls['accuracy']:.4f}",
                f"  macro precision: {cls['macro_precision']:.4f}",
                f"  macro recall: {cls['macro_recall']:.4f}",
                f"  macro F1: {cls['macro_f1']:.4f}",
                f"  weighted F1: {cls['weighted_f1']:.4f}",
                "",
                "Classification report:",
                cls.get("classification_report", ""),
            ]
        )
    lines.extend(["", "Деталь за картинку:"])
    for row in report["per_image"]:
        lines.append(
            f"- {row['image']}: {row['disease']} ({row['confidence']:.3f}) "
            f"[mode={row['mode_used']}, fallback={row['fallback_used']}, hint={row.get('hint_label')}]"
        )
        top3 = ", ".join(f"{t['class']}:{t['confidence']:.2f}" for t in row.get("top3", []))
        lines.append(f"    top3: {top3}")
    path.write_text("\n".join(lines), encoding="utf-8")
