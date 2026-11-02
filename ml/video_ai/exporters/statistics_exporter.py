import json
from pathlib import Path


class StatisticsExporter:
    @staticmethod
    def export(statistics: dict, output_path: str | Path) -> None:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with output_file.open("w", encoding="utf-8") as file:
            json.dump(statistics, file, indent=4)
