from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = Path(os.getenv("VEGVISION_MODELS_DIR", PROJECT_ROOT / "models"))
TEST_IMAGES_DIR = Path(os.getenv("VEGVISION_TEST_IMAGES_DIR", PROJECT_ROOT / "test_images"))
OUTPUTS_DIR = Path(os.getenv("VEGVISION_OUTPUTS_DIR", PROJECT_ROOT / "outputs"))

UNET_CHECKPOINT = Path(
    os.getenv(
        "VEGVISION_UNET_PATH",
        MODELS_DIR / "unet_model.pth",
    )
)
VIT_CHECKPOINT = Path(os.getenv("VEGVISION_VIT_PATH", MODELS_DIR / "vit_model.pth"))

UNET_IMAGE_SIZE = int(os.getenv("VEGVISION_UNET_SIZE", "256"))
VIT_IMAGE_SIZE = int(os.getenv("VEGVISION_VIT_SIZE", "224"))
VIT_RESIZE = int(os.getenv("VEGVISION_VIT_RESIZE", "256"))

SEG_THRESHOLD = float(os.getenv("VEGVISION_SEG_THRESHOLD", "0.5"))
MIN_MASK_AREA_RATIO = float(os.getenv("VEGVISION_MIN_MASK_AREA", "0.01"))
BBOX_PADDING_RATIO = float(os.getenv("VEGVISION_BBOX_PADDING", "0.1"))
CONFIDENCE_THRESHOLD = float(os.getenv("VEGVISION_CONFIDENCE_THRESHOLD", "0.55"))

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEVICE = os.getenv("VEGVISION_DEVICE", "")

EVAL_DIR = OUTPUTS_DIR / "evaluation"
VIS_DIR = OUTPUTS_DIR / "visualizations"
