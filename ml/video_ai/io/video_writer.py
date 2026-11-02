from pathlib import Path

import cv2

from ml.video_ai.config.video_config import VideoConfig
from ml.video_ai.utils.logger import logger


class VideoWriter:
    """Write annotated BGR frames to a video file."""

    def __init__(self, output_path: str, fps: float, width: int, height: int):
        if fps <= 0:
            raise ValueError("fps must be greater than 0")
        if width <= 0 or height <= 0:
            raise ValueError("width and height must be greater than 0")

        self.output_path = output_path
        self.frame_size = (width, height)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*VideoConfig.VIDEO_CODEC)
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, self.frame_size)
        if not self.writer.isOpened():
            raise RuntimeError(f"Could not create video writer: {output_path}")
        logger.info("Video writer created: %s", output_path)

    def write(self, frame) -> None:
        if frame is None or frame.size == 0:
            raise ValueError("Cannot write an empty frame")
        height, width = frame.shape[:2]
        if (width, height) != self.frame_size:
            raise ValueError(
                f"Frame size {(width, height)} does not match writer size {self.frame_size}"
            )
        self.writer.write(frame)

    def release(self) -> None:
        self.writer.release()
        logger.info("Video writer released.")
