"""Run image detection manually: python -m ml.video_ai.test_image_pipeline path/to/image.jpg."""

import sys
from pathlib import Path

from ml.video_ai.pipeline.image_pipeline import ImagePipeline


def main(image_path: str) -> None:
    if not Path(image_path).is_file():
        raise FileNotFoundError(f"Image file not found: {image_path}")

    result = ImagePipeline().process_image(image_path)
    print(f"Objects found: {len(result['detections'])}")
    for detection in result["detections"]:
        print(detection)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python -m ml.video_ai.test_image_pipeline <image_path>")
    main(sys.argv[1])
