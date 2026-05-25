# VegVision

## Ключевые изменения (актуально)

- Публичный endpoint `/predict` больше НЕ принимает внешние `mask`/`bbox` в основном сценарии — API ожидает изображение и опционально `mode` (см. ниже). Внутренний pipeline использует UNet для построения ROI (prob → mask → bbox → full).
- При вызове `/predict` сервис сам запускает сегментацию при необходимости и сохраняет визуализации в `outputs/visualizations/` (поле `visualization_path` в ответе).
---

## Структура проекта (кратко)

```
ml2-tesis-final/
├── app.py                      # FastAPI entrypoint
├── requirements.txt
├── README.md
├── models/                     # веса (обязательны для запуска)
├── test_images/                # локальные тестовые изображения (weak labels по имени файла)
├── scripts/
│   ├── evaluate.py             # offline evaluation (outputs/evaluation/)
│   └── smoke_api.py            # простой smoke-test клиента
├── src/                        # core logic: inference, roi, evaluation, api
└── outputs/                    # outputs/evaluation, outputs/visualizations (runtime)
```

Основная логика inference: `src/inference.py` (включая `predict`) и `src/roi.py` (prepare/фолбэки).

---

## Быстрый старт (локально)

Рекомендовано: Python 3.12 (но Python 3.10+ также поддерживается). Создайте виртуальное окружение и установите зависимости:

```powershell
cd c:\Users\Liza30ku\Desktop\ml2-tesis-final
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Скопируйте/поместите чекпойнты в папку `models/` перед запуском (без них сервис не загрузится).

### Запуск API

```powershell
python app.py
```

Сервер по умолчанию слушает `http://127.0.0.1:8000` (настраивается через `src/settings.py` / env vars).

Для разработки с авто-reload:

```powershell
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

### Offline evaluation

```powershell
python scripts/evaluate.py
```

Результаты сохраняются в `outputs/evaluation/` и `outputs/visualizations/`.

### Smoke-test

Запустите сервер и в другом терминале:

```powershell
python scripts/smoke_api.py
```

---

## API (актуально)

Endpoints:

- `GET /health` — возвращает статус и флаг `models_loaded`.
- `GET /model-info` — информация о моделях, `classes`, `device`, пороги.
- `POST /predict` — основной endpoint; принимает одно изображение и опционально `mode`.
- `POST /predict/batch` — последовательная обработка списка изображений.

POST `/predict` (поля):

- `image` (file) — обязательно.
- `mode` (string) — опционально: `auto` (default), `prob`, `mask`, `bbox`, `full`.

Примечание: публичный контракт больше не подразумевает передачу пользовательских масок/bbox; если вам нужен экспериментальный режим с внешними масками, используйте внутренние утилиты в `src/`.

Пример curl (PowerShell):

```powershell
curl -X POST "http://127.0.0.1:8000/predict" ^
  -F "image=@test_images\Anthracnose (1).jpg" ^
  -F "mode=auto"
```

Пример значимых полей ответа (упрощённо):

```json
{
  "disease": "Cucumber_Bacterial Wilt",
  "confidence": 0.89,
  "confidence_percent": 89,
  "top3": [ ... ],
  "mode_used": "prob",
  "fallback_used": false,
  "needs_agronomist": false,
  "visualization_path": "outputs/predictions/upload_20260523T...png",
  "segmentation": { "has_prob_map": true, "has_mask": true, "has_bbox": true }
}
```

## Переменные окружения и конфигурация

Основные настройки в `src/settings.py`. Важные env vars:

- `VEGVISION_UNET_PATH`, `VEGVISION_VIT_PATH` — пути к чекпойнтам (по умолчанию `models/`).
- `VEGVISION_DEVICE` — `cuda` или `cpu` (автоопределение по умолчанию).
- `VEGVISION_CONFIDENCE_THRESHOLD` — порог для `needs_agronomist` (дефолт ~0.55).
- `VEGVISION_HOST`, `VEGVISION_PORT` — bind для API.
