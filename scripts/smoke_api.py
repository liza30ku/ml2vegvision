from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
TEST_IMAGES = ROOT / "test_images"
BASE_URL = "http://127.0.0.1:8000"


def main() -> int:
    images = sorted(
        p
        for p in TEST_IMAGES.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    )
    if not images:
        print("Нет изображений в test_images")
        return 1

    with httpx.Client(base_url=BASE_URL, timeout=120.0) as client:
        health = client.get("/health")
        health.raise_for_status()
        print("GET /health ->", health.json())

        info = client.get("/model-info")
        info.raise_for_status()
        meta = info.json()
        print(f"GET /model-info -> {meta['num_classes']} clases, device={meta['device']}")

        sample = images[0]
        with sample.open("rb") as f:
            resp = client.post(
                "/predict",
                files={"image": (sample.name, f, "image/jpeg")},
                data={"mode": "auto"},
            )
        resp.raise_for_status()
        one = resp.json()
        print(
            f"POST /predict ({sample.name}) -> {one['disease']} "
            f"conf={one['confidence']:.3f} mode={one['mode_used']}"
        )

        files = [("images", (p.name, p.open("rb"), "image/jpeg")) for p in images[:3]]
        try:
            batch = client.post("/predict/batch", files=files, data={"mode": "auto"})
            batch.raise_for_status()
            payload = batch.json()
            ok = sum(1 for r in payload["results"] if r["success"])
            print(f"POST /predict/batch -> {ok}/{payload['count']} ok")
            for row in payload["results"]:
                if row["success"]:
                    r = row["result"]
                    print(f"  - {row['filename']}: {r['disease']} ({r['confidence']:.3f})")
                else:
                    print(f"  - {row['filename']}: ERROR {row['error']}")
        finally:
            for _, (_, fh, _) in files:
                fh.close()

    print("Smoke test OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except httpx.HTTPError as exc:
        print("Smoke test прошел с ошибками:", exc)
        print("Убедитесь, что сервер запущен: python app.py")
        raise SystemExit(1) from exc
