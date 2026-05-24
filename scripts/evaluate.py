from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation import run_test_images_evaluation
from src.settings import EVAL_DIR, TEST_IMAGES_DIR


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка моделей VegVision на test_images")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=TEST_IMAGES_DIR,
        help="Директория с изображениями для тестирования",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=EVAL_DIR,
        help="Директория выхода отчета",
    )
    args = parser.parse_args()
    report = run_test_images_evaluation(args.images_dir, args.output_dir)
    print(f"Оценка завершена: {args.output_dir / 'evaluation_report.json'}")
    print(f"Resumen: {args.output_dir / 'summary.txt'}")
    print(f"Modos: {report['metrics']['mode_distribution']}")


if __name__ == "__main__":
    main()
