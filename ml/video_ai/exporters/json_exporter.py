import json
from pathlib import Path


class JsonExporter:

    @staticmethod
    def export(
        detections,
        output_path: str,
        mode: str = "image"
    ):
        """
        Export detections to JSON.

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
            Output JSON file path.

        mode : str
            "image" or "video"
        """

        # ==========================================
        # Validate Mode
        # ==========================================

        if mode not in ("image", "video"):

            raise ValueError(
                "mode must be either "
                "'image' or 'video'"
            )

        data = []

        # ==========================================
        # IMAGE
        # ==========================================

        if mode == "image":

            for detection in detections:

                data.append({

                    "class_id":
                        detection.class_id,

                    "class_name":
                        detection.class_name,

                    "confidence":
                        round(
                            detection.confidence,
                            4
                        ),

                    "bbox": {

                        "x1":
                            detection.bbox.x1,

                        "y1":
                            detection.bbox.y1,

                        "x2":
                            detection.bbox.x2,

                        "y2":
                            detection.bbox.y2
                    }
                })

        # ==========================================
        # VIDEO
        # ==========================================

        elif mode == "video":

            for item in detections:

                frame_number = (
                    item["frame_number"]
                )

                detection = (
                    item["detection"]
                )

                data.append({

                    "frame_number":
                        frame_number,

                    "class_id":
                        detection.class_id,

                    "class_name":
                        detection.class_name,

                    "confidence":
                        round(
                            detection.confidence,
                            4
                        ),

                    "bbox": {

                        "x1":
                            detection.bbox.x1,

                        "y1":
                            detection.bbox.y1,

                        "x2":
                            detection.bbox.x2,

                        "y2":
                            detection.bbox.y2
                    }
                })

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
        # WRITE JSON
        # ==========================================

        with open(
            output_file,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=4
            )