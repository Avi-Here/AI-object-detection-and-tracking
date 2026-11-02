import time
from collections import Counter

from ml.video_ai.config.video_config import VideoConfig
from ml.video_ai.exporters.csv_exporter import CsvExporter
from ml.video_ai.exporters.json_exporter import JsonExporter
from ml.video_ai.exporters.statistics_exporter import StatisticsExporter
from ml.video_ai.exporters.timeline_exporter import TimelineExporter
from ml.video_ai.io.video_reader import VideoReader
from ml.video_ai.io.video_writer import VideoWriter
from ml.video_ai.pipeline.image_pipeline import ImagePipeline
from ml.video_ai.utils.logger import logger


class VideoPipeline:
    """Process video frames with YOLO detection and optional object tracking."""

    def __init__(self):
        logger.info("Creating VideoPipeline...")
        self.image_pipeline = ImagePipeline()
        logger.info("VideoPipeline created successfully.")

    def process(self, video_path: str) -> dict:
        """Process a video and return detection statistics and its timeline."""
        if VideoConfig.FRAME_SKIP < 1:
            raise ValueError("FRAME_SKIP must be at least 1")

        start_time = time.time()
        reader = VideoReader(video_path)
        writer = None

        try:
            fps = reader.get_fps()
            total_frames = reader.get_frame_count()
            width = reader.get_width()
            height = reader.get_height()
            if VideoConfig.SAVE_VIDEO:
                writer = VideoWriter(VideoConfig.OUTPUT_VIDEO, fps, width, height)

            logger.info(
                "Processing %s | %.2f FPS | %s frames | %sx%s",
                video_path,
                fps,
                total_frames,
                width,
                height,
            )

            processed_frames = 0
            total_detections = 0
            detections_by_class = Counter()
            tracked_object_keys: set[tuple[str, int]] = set()
            unique_objects_by_class = Counter()
            all_detections = []
            timeline = []

            for frame_number, frame in self._frames(reader):
                should_detect = (frame_number - 1) % VideoConfig.FRAME_SKIP == 0
                output_frame = frame
                frame_objects = []

                if should_detect:
                    result = self.image_pipeline.process_frame(
                        frame, tracking=VideoConfig.ENABLE_TRACKING
                    )
                    output_frame = result["frame"]
                    detections = result["detections"]
                    processed_frames += 1
                    total_detections += len(detections)

                    for detection in detections:
                        detections_by_class[detection.class_name] += 1
                        frame_objects.append(self._timeline_object(detection))
                        all_detections.append(
                            {"frame_number": frame_number, "detection": detection}
                        )

                        if detection.track_id is not None:
                            key = (detection.class_name, detection.track_id)
                            if key not in tracked_object_keys:
                                tracked_object_keys.add(key)
                                unique_objects_by_class[detection.class_name] += 1

                timeline.append(
                    {
                        "frame_number": frame_number,
                        "processed": should_detect,
                        "objects": frame_objects,
                    }
                )

                # Write every frame so frame skipping never shortens the video.
                if writer is not None:
                    writer.write(output_frame)

                if frame_number % 10 == 0 or frame_number == total_frames:
                    logger.info("Frame %s/%s", frame_number, total_frames)
        finally:
            reader.release()
            if writer is not None:
                writer.release()

        statistics = {
            "video_path": video_path,
            "output_video": VideoConfig.OUTPUT_VIDEO if VideoConfig.SAVE_VIDEO else None,
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "total_detections": total_detections,
            "detections_by_class": dict(detections_by_class),
            "unique_objects": len(tracked_object_keys),
            "unique_objects_by_class": dict(unique_objects_by_class),
            "tracking_enabled": VideoConfig.ENABLE_TRACKING,
            "processing_time": round(time.time() - start_time, 2),
        }

        self._export_outputs(all_detections, statistics, timeline)
        logger.info("Video processing completed in %.2f seconds.", statistics["processing_time"])
        return {"statistics": statistics, "timeline": timeline}

    @staticmethod
    def _frames(reader):
        """Yield source frames with one-based frame numbers."""
        frame_number = 0
        while True:
            success, frame = reader.read()
            if not success:
                return
            frame_number += 1
            yield frame_number, frame

    @staticmethod
    def _timeline_object(detection) -> dict:
        return {
            "class_name": detection.class_name,
            "track_id": detection.track_id,
            "confidence": round(detection.confidence, 4),
            "bbox": {
                "x1": detection.bbox.x1,
                "y1": detection.bbox.y1,
                "x2": detection.bbox.x2,
                "y2": detection.bbox.y2,
            },
        }

    @staticmethod
    def _export_outputs(all_detections: list, statistics: dict, timeline: list) -> None:
        if VideoConfig.SAVE_JSON:
            JsonExporter.export(all_detections, VideoConfig.OUTPUT_JSON, mode="video")
        if VideoConfig.SAVE_CSV:
            CsvExporter.export(all_detections, VideoConfig.OUTPUT_CSV, mode="video")
        if VideoConfig.SAVE_STATISTICS:
            StatisticsExporter.export(statistics, VideoConfig.OUTPUT_STATISTICS)
        if VideoConfig.SAVE_TIMELINE:
            TimelineExporter.export(timeline, VideoConfig.OUTPUT_TIMELINE)
