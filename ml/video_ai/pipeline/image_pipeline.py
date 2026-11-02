from pathlib import Path

import cv2
import numpy as np

from ml.video_ai.config.video_config import VideoConfig
from ml.video_ai.detection.object_detector import ObjectDetector

from ml.video_ai.exporters.csv_exporter import CsvExporter
from ml.video_ai.exporters.json_exporter import JsonExporter

from ml.video_ai.utils.drawing import draw_detections
from ml.video_ai.utils.logger import logger


class ImagePipeline:
    """
    Handles image and frame processing.

    The pipeline does not know whether the frame
    came from an image, video, webcam, etc.

    Responsibilities:
    - Run object detection.
    - Optionally run tracking.
    - Draw detections.
    - Return detections and annotated frame.
    - Save outputs for standalone image processing.
    """

    def __init__(self):

        logger.info(
            "Creating ImagePipeline..."
        )

        # ==========================================
        # Object Detector
        # ==========================================

        # Created only ONCE.
        #
        # ObjectDetector loads YOLO model.
        #
        # Therefore YOLO model is also loaded only ONCE.

        self.object_detector = ObjectDetector()

        logger.info(
            "ImagePipeline created successfully."
        )

    # ==========================================
    # IMAGE PROCESSING
    # ==========================================

    def process_image(
        self,
        image_path: str
    ):
        """
        Process a complete image.

        Steps:

        Image
           ↓
        Read Image
           ↓
        process_frame()
           ↓
        Detection
           ↓
        Drawing
           ↓
        Save Image / JSON / CSV
        """

        logger.info(
            f"Reading image: {image_path}"
        )

        # ------------------------------------------
        # Read Image
        # ------------------------------------------

        image = cv2.imread(
            image_path
        )

        if image is None:

            raise FileNotFoundError(
                f"Could not read image: {image_path}"
            )

        # ------------------------------------------
        # Process Frame
        # ------------------------------------------

        result = self.process_frame(
            image
        )

        # ------------------------------------------
        # Save Image Outputs
        # ------------------------------------------

        self.save_image_outputs(
            result["frame"],
            result["detections"]
        )

        return result

    # ==========================================
    # FRAME PROCESSING
    # ==========================================

    def process_frame(
        self,
        frame: np.ndarray,
        tracking: bool = False
    ):
        """
        Process a single frame.

        This method can be used by:

        - ImagePipeline
        - VideoPipeline
        - WebcamPipeline

        Parameters
        ----------
        frame : np.ndarray
            Image/frame to process.

        tracking : bool
            If True, tracking will be used.
            If False, normal object detection will be used.

        Returns
        -------
        dict
            {
                "frame": annotated_frame,
                "detections": detections
            }
        """

        # ==========================================
        # Detection / Tracking
        # ==========================================

        if tracking:

            detections = (
                self.object_detector.track(
                    frame
                )
            )

        else:

            detections = (
                self.object_detector.detect(
                    frame
                )
            )

        logger.info(
            f"Detected {len(detections)} object(s)."
        )

        # ==========================================
        # Draw Detections
        # ==========================================

        annotated_frame = draw_detections(
            frame.copy(),
            detections
        )

        # ==========================================
        # Return Result
        # ==========================================

        return {
            "frame": annotated_frame,
            "detections": detections
        }

    # ==========================================
    # IMAGE OUTPUTS
    # ==========================================

    def save_image_outputs(
        self,
        image,
        detections
    ):
        """
        Save outputs when processing
        a standalone image.

        VideoPipeline does NOT call this method.
        """

        # ==========================================
        # Save Annotated Image
        # ==========================================

        if VideoConfig.SAVE_ANNOTATED_IMAGE:

            output_path = Path(
                VideoConfig.OUTPUT_IMAGE
            )

            output_path.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            cv2.imwrite(
                str(output_path),
                image
            )

            logger.info(
                f"Annotated image saved to "
                f"{output_path}"
            )

        # ==========================================
        # Save JSON
        # ==========================================

        if VideoConfig.SAVE_JSON:

            JsonExporter.export(
                detections,
                VideoConfig.OUTPUT_JSON,
                mode="image"
            )

            logger.info(
                "Image detection JSON exported."
            )

        # ==========================================
        # Save CSV
        # ==========================================

        if VideoConfig.SAVE_CSV:

            CsvExporter.export(
                detections,
                VideoConfig.OUTPUT_CSV,
                mode="image"
            )

            logger.info(
                "Image detection CSV exported."
            )

        logger.info(
            "Image outputs saved successfully."
        )