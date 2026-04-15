from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field


class StatementCreate(BaseModel):
    role: str
    vehicle: Optional[str] = None
    text: str

    class Config:
        json_schema_extra = {
            "example": {
                "role": "driver",
                "vehicle": "Toyota Corolla 2017",
                "text": "A car hit me at the roundabout and left the scene.",
            }
        }


class CaseCreate(BaseModel):
    claim_id: str
    location: str
    incident_type: str
    detected_damages: List[str] = Field(default_factory=list)
    statements: List[StatementCreate] = Field(default_factory=list)

    class Config:
        json_schema_extra = {
            "example": {
                "claim_id": "PT-GOOD-2026-002007",
                "location": "Avenida da Republica, Lisbon",
                "incident_type": "rear-end collision",
                "detected_damages": ["rear bumper dent", "tail light crack"],
                "statements": [
                    {
                        "role": "driver",
                        "vehicle": "Toyota Corolla 2017",
                        "text": "I was stopped at the light when I felt impact.",
                    },
                    {
                        "role": "passenger",
                        "vehicle": "Toyota Corolla 2017",
                        "text": "We were stationary and got hit from behind.",
                    },
                ],
            }
        }


class StatementRead(BaseModel):
    id: int
    role: str
    vehicle: Optional[str]
    text: str

    class Config:
        from_attributes = True


class ImageRead(BaseModel):
    id: int
    statement_id: Optional[int]
    file_path: str
    damage_types: Optional[List[str]]
    yolo_raw_output: Optional[dict]

    class Config:
        from_attributes = True


class CaseRead(BaseModel):
    id: int
    claim_id: str
    location: str
    incident_type: str
    detected_damages: List[str]
    created_at: datetime
    probability_true: Optional[float]
    verdict: Optional[str]
    reasoning: Optional[str]
    incongruences: Optional[List[str]]
    qwen_raw_output: Optional[str]
    statements: List[StatementRead]
    images: List[ImageRead]

    class Config:
        from_attributes = True


class AnalyzeResponse(BaseModel):
    case: CaseRead
    qwen_schema_valid: bool
    qwen_validation_errors: List[str]
    qwen_parse_error: Optional[str]
