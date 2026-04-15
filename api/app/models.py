from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from sqlalchemy import DateTime, ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.types import JSON


class Base(DeclarativeBase):
    pass


class Case(Base):
    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(primary_key=True)
    claim_id: Mapped[str] = mapped_column(String(100))
    location: Mapped[str] = mapped_column(String(255))
    incident_type: Mapped[str] = mapped_column(String(255))
    detected_damages: Mapped[list] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    qwen_raw_output: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    probability_true: Mapped[Optional[float]] = mapped_column(nullable=True)
    verdict: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)
    reasoning: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    incongruences: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    statements: Mapped[List["Statement"]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan",
    )
    images: Mapped[List["Image"]] = relationship(
        back_populates="case",
        cascade="all, delete-orphan",
    )


class Statement(Base):
    __tablename__ = "statements"

    id: Mapped[int] = mapped_column(primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"))
    role: Mapped[str] = mapped_column(String(120))
    vehicle: Mapped[Optional[str]] = mapped_column(String(120), nullable=True)
    text: Mapped[str] = mapped_column(String)

    case: Mapped[Case] = relationship(back_populates="statements")
    images: Mapped[List["Image"]] = relationship(back_populates="statement")


class Image(Base):
    __tablename__ = "images"

    id: Mapped[int] = mapped_column(primary_key=True)
    case_id: Mapped[int] = mapped_column(ForeignKey("cases.id"))
    statement_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("statements.id"),
        nullable=True,
    )
    file_path: Mapped[str] = mapped_column(String(500))
    yolo_raw_output: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    damage_types: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    case: Mapped[Case] = relationship(back_populates="images")
    statement: Mapped[Optional[Statement]] = relationship(back_populates="images")
