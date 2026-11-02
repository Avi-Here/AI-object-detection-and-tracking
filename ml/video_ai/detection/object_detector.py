import numpy as np
from ultralytics import YOLO

from ml.video_ai.config.video_config import VideoConfig
from ml.video_ai.entities.bounding_box import BoundingBox
from ml.video_ai.entities.detection import Detection
from ml.video_ai.utils.logger import logger


class ObjectDetector:
    """Load YOLO once and expose detection and persistent tracking methods."""

    def __init__(self):
        logger.info("Loading YOLO model: %s", VideoConfig.MODEL_NAME)
        self.model = YOLO(VideoConfig.MODEL_NAME)
        logger.info("YOLO model loaded successfully.")

    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect objects in an independent image/frame."""
        results = self.model.predict(
            source=frame,
            conf=VideoConfig.CONFIDENCE_THRESHOLD,
            imgsz=VideoConfig.IMAGE_SIZE,
            device=VideoConfig.DEVICE,
            verbose=False,
        )
        return self.parse_results(results)

    def track(self, frame: np.ndarray) -> list[Detection]:
        """Track objects in sequential video frames using the configured tracker."""
        results = self.model.track(
            source=frame,
            conf=VideoConfig.CONFIDENCE_THRESHOLD,
            imgsz=VideoConfig.IMAGE_SIZE,
            device=VideoConfig.DEVICE,
            tracker=VideoConfig.TRACKER,
            persist=True,
            verbose=False,
        )
        return self.parse_results(results, tracking=True)

    @staticmethod
    def parse_results(results, *, tracking: bool = False) -> list[Detection]:
        """Convert Ultralytics results to application detection objects."""
        if not results:
            return []

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []

        detections = []
        for box in result.boxes:
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            track_id = int(box.id.item()) if tracking and box.id is not None else None
            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=result.names[class_id],
                    confidence=float(box.conf.item()),
                    bbox=BoundingBox(int(x1), int(y1), int(x2), int(y2)),
                    track_id=track_id,
                )
            )
        return detections
