import numpy as np

from ml.video_ai.entities.bounding_box import BoundingBox
from ml.video_ai.entities.detection import Detection
from ml.video_ai.utils.drawing import draw_detections


def test_draw_detections_adds_annotation() -> None:
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    detections = [
        Detection(
            class_id=0,
            class_name="person",
            confidence=0.95,
            bbox=BoundingBox(10, 20, 50, 70),
            track_id=1,
        )
    ]

    result = draw_detections(image, detections)

    assert result.shape == image.shape
    assert np.any(result != 0)
