from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from src.api.routes import router
from src.services.inference_service import ensure_models_loaded

logging.basicConfig(
    level=os.getenv("VEGVISION_LOG_LEVEL", "INFO"),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(_app: FastAPI):
    logger.info("Загрузка моделей при запуске...")
    ensure_models_loaded()
    logger.info("Модели готовы.")
    yield


app = FastAPI(
    title="VegVision API",
    description="Baseline inference для диагностики болезней у растений.",
    version="0.1.0",
    lifespan=lifespan,
)
app.include_router(router)


if __name__ == "__main__":
    host = os.getenv("VEGVISION_HOST", "0.0.0.0")
    port = int(os.getenv("VEGVISION_PORT", "8000"))
    uvicorn.run("app:app", host=host, port=port, reload=False)
