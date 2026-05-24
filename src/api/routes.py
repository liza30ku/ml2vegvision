from __future__ import annotations

import logging

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from src.schemas.predict import (
    BatchItemResult,
    BatchPredictResponse,
    HealthResponse,
    ModelInfoResponse,
    PredictResponse,
)
from src.services.inference_service import (
    get_model_info,
    models_loaded,
    run_predict,
)

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", models_loaded=models_loaded())


@router.get("/model-info", response_model=ModelInfoResponse)
def model_info() -> ModelInfoResponse:
    try:
        return ModelInfoResponse(**get_model_info())
    except Exception as exc:
        logger.exception("Ошибка загрузки модели инфо")
        raise HTTPException(status_code=500, detail=f"No se pudo cargar informacion del modelo: {exc}") from exc


@router.post("/predict", response_model=PredictResponse)
async def predict_endpoint(
    image: UploadFile = File(..., description="Изображение листа (обязательно)"),
    mode: str | None = Form("auto", description="auto | prob | mask | bbox | full"),
) -> PredictResponse:
    if not image.filename:
        raise HTTPException(status_code=422, detail="Требуется файл изображения")
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Изображение пусто")

    try:
        result = run_predict(image_bytes, mode=mode, original_filename=image.filename)
        return PredictResponse(**result)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ошибка в /predict")
        raise HTTPException(status_code=500, detail=f"Ошибка инференса: {exc}") from exc


@router.post("/predict/batch", response_model=BatchPredictResponse)
async def predict_batch_endpoint(
    images: list[UploadFile] = File(..., description="Список изображений"),
    mode: str | None = Form("auto"),
) -> BatchPredictResponse:
    if not images:
        raise HTTPException(status_code=422, detail="Нужно загрузить хотя бы одно изображение")

    results: list[BatchItemResult] = []
    for upload in images:
        name = upload.filename or "unknown"
        try:
            data = await upload.read()
            if not data:
                raise ValueError("изображение пусто")
            payload = run_predict(data, mode=mode)
            results.append(
                BatchItemResult(filename=name, success=True, result=PredictResponse(**payload))
            )
        except ValueError as exc:
            results.append(BatchItemResult(filename=name, success=False, error=str(exc)))
        except Exception as exc:
            logger.exception("Ошибка в batch для %s", name)
            results.append(BatchItemResult(filename=name, success=False, error=str(exc)))

    return BatchPredictResponse(count=len(results), results=results)
