from __future__ import annotations

import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile

from .config import settings
from .db import get_session
from .models import Base, Case, Image, Statement
from .schemas import AnalyzeResponse, CaseCreate, CaseRead, ImageRead, StatementRead
from .services.qwen_service import run_inference
from .services.yolo_service import YoloRunner

from sqlalchemy.orm import Session
from sqlalchemy import select
from .db import engine


app = FastAPI(
    title=settings.app_name,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


@app.on_event("startup")
def startup() -> None:
    Base.metadata.create_all(bind=engine)
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)


@app.post("/cases", response_model=CaseRead)
def create_case(
    payload: CaseCreate, session: Session = Depends(get_session)
) -> CaseRead:
    case = Case(
        claim_id=payload.claim_id,
        location=payload.location,
        incident_type=payload.incident_type,
        detected_damages=payload.detected_damages,
    )
    session.add(case)
    session.flush()
    for statement in payload.statements:
        session.add(
            Statement(
                case_id=case.id,
                role=statement.role,
                vehicle=statement.vehicle,
                text=statement.text,
            )
        )
    session.flush()
    session.refresh(case)
    return CaseRead.model_validate(case)


@app.post("/cases/{case_id}/images", response_model=List[ImageRead])
def upload_images(
    case_id: int,
    files: List[UploadFile] = File(...),
    statement_id: Optional[int] = Query(
        default=None,
        description="Optional statement id to associate the image",
        example=1,
    ),
    session: Session = Depends(get_session),
) -> List[ImageRead]:
    case = session.get(Case, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    if statement_id is not None:
        statement = session.get(Statement, statement_id)
        if statement is None or statement.case_id != case_id:
            raise HTTPException(status_code=400, detail="Invalid statement_id")

    yolo_runner = YoloRunner(settings.yolo_weights_path, settings.yolo_confidence)
    stored_images: List[ImageRead] = []

    for file in files:
        filename = f"case_{case_id}_{file.filename}"
        file_path = settings.uploads_dir / filename
        with file_path.open("wb") as handle:
            shutil.copyfileobj(file.file, handle)
        damage_types, raw_output = yolo_runner.predict(file_path)
        image = Image(
            case_id=case_id,
            statement_id=statement_id,
            file_path=str(file_path),
            damage_types=damage_types,
            yolo_raw_output=raw_output,
        )
        session.add(image)
        session.flush()
        stored_images.append(ImageRead.model_validate(image))

    return stored_images


@app.post("/cases/{case_id}/analyze", response_model=AnalyzeResponse)
def analyze_case(
    case_id: int, session: Session = Depends(get_session)
) -> AnalyzeResponse:
    case = session.get(Case, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")

    statements = [
        {"role": st.role, "vehicle": st.vehicle or "", "text": st.text}
        for st in case.statements
    ]
    raw_output, parsed, parse_error, schema_valid, validation_errors = run_inference(
        model_id=settings.qwen_model_id,
        adapter_path=str(settings.qwen_adapter_path)
        if settings.qwen_adapter_path
        else None,
        max_tokens=settings.qwen_max_tokens,
        temperature=settings.qwen_temperature,
        top_p=settings.qwen_top_p,
        claim_id=case.claim_id,
        location=case.location,
        incident_type=case.incident_type,
        detected_damages=case.detected_damages,
        statements=statements,
    )
    case.qwen_raw_output = raw_output
    if isinstance(parsed, dict):
        case.probability_true = parsed.get("probability_true")
        case.verdict = parsed.get("verdict")
        case.reasoning = parsed.get("reasoning")
        case.incongruences = parsed.get("incongruences")
    session.add(case)
    session.flush()
    session.refresh(case)

    return AnalyzeResponse(
        case=CaseRead.model_validate(case),
        qwen_schema_valid=schema_valid,
        qwen_validation_errors=validation_errors,
        qwen_parse_error=parse_error,
    )


@app.get("/cases/{case_id}", response_model=CaseRead)
def get_case(case_id: int, session: Session = Depends(get_session)) -> CaseRead:
    case = session.get(Case, case_id)
    if case is None:
        raise HTTPException(status_code=404, detail="Case not found")
    return CaseRead.model_validate(case)


@app.get("/cases", response_model=List[CaseRead])
def list_cases(session: Session = Depends(get_session)) -> List[CaseRead]:
    cases = session.scalars(select(Case).order_by(Case.created_at.desc())).all()
    return [CaseRead.model_validate(case) for case in cases]
