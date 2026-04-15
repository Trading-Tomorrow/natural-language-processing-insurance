from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

from ultralytics import YOLO


class YoloRunner:
    def __init__(self, weights_path: Path, confidence: float) -> None:
        self.model = YOLO(str(weights_path))
        self.confidence = confidence

    def predict(self, image_path: Path) -> Tuple[List[str], Dict[str, Any]]:
        results = self.model.predict(
            source=str(image_path),
            conf=self.confidence,
            save=False,
            show=False,
        )
        detections: List[Dict[str, Any]] = []
        damage_types: List[str] = []
        for result in results:
            names = result.names or {}
            boxes = result.boxes
            if boxes is None:
                continue
            for box in boxes:
                cls_index = int(box.cls.item()) if box.cls is not None else None
                conf = float(box.conf.item()) if box.conf is not None else None
                xyxy = box.xyxy[0].tolist() if box.xyxy is not None else None
                cls_name = (
                    names.get(cls_index, str(cls_index))
                    if cls_index is not None
                    else None
                )
                if cls_name is not None:
                    damage_types.append(cls_name)
                detections.append(
                    {
                        "class": cls_name,
                        "confidence": conf,
                        "box": xyxy,
                    }
                )
        raw_output = {
            "detections": detections,
        }
        return damage_types, raw_output
