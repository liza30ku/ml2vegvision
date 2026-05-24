from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class TopClassItem(BaseModel):
    class_name: str = Field(alias="class")
    confidence: float

    model_config = {"populate_by_name": True}


class SegmentationInfo(BaseModel):
    has_prob_map: bool
    has_mask: bool
    has_bbox: bool


class PredictResponse(BaseModel):
    disease: str
    confidence: float
    confidence_percent: int
    top3: list[TopClassItem]
    mode_requested: str
    mode_used: str
    fallback_used: bool
    bbox_used: list[int] | None = None
    mask_used: bool
    prob_used: bool
    needs_agronomist: bool
    message: str
    image_size: list[int]
    segmentation: SegmentationInfo
    visualization_path: str | None = None
    visualization_created: bool = False


class BatchItemResult(BaseModel):
    filename: str
    success: bool
    result: PredictResponse | None = None
    error: str | None = None


class BatchPredictResponse(BaseModel):
    count: int
    results: list[BatchItemResult]


class ModelInfoResponse(BaseModel):
    segmentation_model: str
    classifier_model: str
    num_classes: int
    classes: dict[int, str]
    device: str
    vit_best_val_acc: float | None = None
    confidence_threshold: float
    roi_modes: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    models_loaded: bool
