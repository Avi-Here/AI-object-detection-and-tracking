import csv
from pathlib import Path


class CsvExporter:

    @staticmethod
    def export(
        detections,
        output_path: str,
        mode: str = "image"
    ):
        """
        Export detections to CSV.

        Image mode:
            detections = list[Detection]

        Video mode:
            detections = [
                {
                    "frame_number": int,
                    "detection": Detection
                }
            ]

        Parameters
        ----------
        detections : list
            Detection data.

        output_path : str
            Path of the output CSV file.

        mode : str
            "image" or "video"
        """

        # ==========================================
        # VALIDATE MODE
        # ==========================================

        if mode not in ("image", "video"):

            raise ValueError(
                "mode must be either "
                "'image' or 'video'"
            )

        # ==========================================
        # CREATE OUTPUT DIRECTORY
        # ==========================================

        output_file = Path(
            output_path
        )

        output_file.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        # ==========================================
        # OPEN CSV FILE
        # ==========================================

        with open(
            output_file,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.writer(file)

            # ======================================
            # IMAGE
            # ======================================

            if mode == "image":

                writer.writerow([
                    "class_id",
                    "class_name",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2"
                ])

                for detection in detections:

                    writer.writerow([

                        detection.class_id,

                        detection.class_name,

                        round(
                            detection.confidence,
                            4
                        ),

                        detection.bbox.x1,
                        detection.bbox.y1,
                        detection.bbox.x2,
                        detection.bbox.y2
                    ])

            # ======================================
            # VIDEO
            # ======================================

            elif mode == "video":

                writer.writerow([
                    "frame_number",
                    "class_id",
                    "class_name",
                    "confidence",
                    "x1",
                    "y1",
                    "x2",
                    "y2"
                ])

                for item in detections:

                    frame_number = (
                        item["frame_number"]
                    )

                    detection = (
                        item["detection"]
                    )

                    writer.writerow([

                        frame_number,

                        detection.class_id,

                        detection.class_name,

                        round(
                            detection.confidence,
                            4
                        ),

                        detection.bbox.x1,
                        detection.bbox.y1,
                        detection.bbox.x2,
                        detection.bbox.y2
                    ])