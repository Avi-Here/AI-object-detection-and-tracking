import json
from pathlib import Path


class TimelineExporter:
    @staticmethod
    def export(timeline: list, output_path: str | Path) -> None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as file:
            json.dump(timeline, file, indent=4)
