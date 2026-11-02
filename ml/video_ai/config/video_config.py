class VideoConfig:
    """Configuration settings for the Video AI module."""

    # YOLO
    MODEL_NAME = "yolo11n.pt"
    CONFIDENCE_THRESHOLD = 0.50
    IMAGE_SIZE = 640
    DEVICE = "cpu"

    # Tracking
    ENABLE_TRACKING = True
    TRACKER = "bytetrack.yaml"

    # Drawing
    BOX_COLOR = (0, 255, 0)
    BOX_THICKNESS = 2
    FONT_SCALE = 0.6
    FONT_THICKNESS = 2

    # Image output
    SAVE_ANNOTATED_IMAGE = True
    OUTPUT_IMAGE = "ml/video_ai/outputs/images/output.jpg"

    # Detection output
    SAVE_JSON = True
    SAVE_CSV = True
    OUTPUT_JSON = "ml/video_ai/outputs/json/detections.json"
    OUTPUT_CSV = "ml/video_ai/outputs/csv/detections.csv"

    # Video output
    SAVE_VIDEO = True
    OUTPUT_VIDEO = "ml/video_ai/outputs/videos/output.mp4"
    VIDEO_CODEC = "mp4v"

    # Video processing
    FRAME_SKIP = 1

    # Analytics output
    SAVE_STATISTICS = True
    OUTPUT_STATISTICS = "ml/video_ai/outputs/json/video_statistics.json"
    SAVE_TIMELINE = True
    OUTPUT_TIMELINE = "ml/video_ai/outputs/json/frame_timeline.json"
