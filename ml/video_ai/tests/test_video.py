from pathlib import Path

from ml.video_ai.pipeline.video_pipeline import VideoPipeline


VIDEO_PATH = "ml/video_ai/inputs/sample.mp4"


def main() -> None:
    if not Path(VIDEO_PATH).is_file():
        raise FileNotFoundError(f"Video file not found: {VIDEO_PATH}")

    result = VideoPipeline().process(VIDEO_PATH)
    statistics = result["statistics"]

    print("\nVIDEO PROCESSING COMPLETED")
    print("Output Video:", statistics["output_video"])
    print("Total Frames:", statistics["total_frames"])
    print("Processed Frames:", statistics["processed_frames"])
    print("Total Detections:", statistics["total_detections"])
    print("Unique Objects:", statistics["unique_objects"])
    print("Unique Objects By Class:", statistics["unique_objects_by_class"])


if __name__ == "__main__":
    main()
