import cv2

from ml.video_ai.config.video_config import VideoConfig
from ml.video_ai.entities.detection import Detection


def draw_detections(
    image,
    detections: list[Detection]
):
    """
    Draw bounding boxes and labels on the image.

    Parameters
    ----------
    image : np.ndarray
        Input image/frame.

    detections : list[Detection]
        List of Detection objects.

    Returns
    -------
    np.ndarray
        Annotated image/frame.
    """

    for detection in detections:

        # ==========================================
        # Bounding Box
        # ==========================================

        bbox = detection.bbox

        x1 = bbox.x1
        y1 = bbox.y1
        x2 = bbox.x2
        y2 = bbox.y2

        # ==========================================
        # Detection Information
        # ==========================================

        class_name = detection.class_name

        confidence = detection.confidence

        # ==========================================
        # Label
        # ==========================================

        label = (
            f"{class_name} "
            f"({confidence:.2f})"
        )

        # ==========================================
        # Draw Bounding Box
        # ==========================================

        cv2.rectangle(
            image,
            (x1, y1),
            (x2, y2),
            VideoConfig.BOX_COLOR,
            VideoConfig.BOX_THICKNESS
        )

        # ==========================================
        # Draw Label
        # ==========================================

        cv2.putText(
            image,
            label,
            (
                x1,
                max(y1 - 10, 20)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            VideoConfig.FONT_SCALE,
            VideoConfig.BOX_COLOR,
            VideoConfig.FONT_THICKNESS
        )

    return image