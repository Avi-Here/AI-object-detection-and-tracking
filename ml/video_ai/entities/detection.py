from dataclasses import dataclass

from ml.video_ai.entities.bounding_box import BoundingBox


@dataclass
class Detection:
    """
    Represents one detected object.
    """

    class_id: int
    class_name: str
    confidence: float
    bbox: BoundingBox
    # None for normal image detection.
    track_id: int | None = None
