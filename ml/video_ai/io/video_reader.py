import cv2

from ml.video_ai.utils.logger import logger


class VideoReader:
    """Read frames and validated metadata from a video file."""

    def __init__(self, video_path: str):
        self.video_path = video_path
        self.capture = cv2.VideoCapture(video_path)
        if not self.capture.isOpened():
            self.capture.release()
            raise FileNotFoundError(f"Could not open video: {video_path}")
        logger.info("Video opened successfully: %s", video_path)

    def get_fps(self) -> float:
        fps = self.capture.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            raise ValueError(f"Video has an invalid FPS value: {fps}")
        return fps

    def get_frame_count(self) -> int:
        return int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def get_width(self) -> int:
        width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        if width <= 0:
            raise ValueError(f"Video has an invalid width value: {width}")
        return width

    def get_height(self) -> int:
        height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if height <= 0:
            raise ValueError(f"Video has an invalid height value: {height}")
        return height

    def read(self):
        return self.capture.read()

    def release(self) -> None:
        self.capture.release()
        logger.info("Video reader released.")
