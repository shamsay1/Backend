from sqlalchemy import (
    Column,
    Integer,
    Float,
    Text,
    ForeignKey,
)
from sqlalchemy.orm import relationship

from .database import Base


class AyahSegment(Base):

    __tablename__ = "ayah_segments"

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    ayah_id = Column(
        Integer,
        ForeignKey("ayah.id", ondelete="CASCADE"),
        nullable=False,
        index=True
    )

    segment_number = Column(
        Integer,
        nullable=False
    )

    text = Column(
        Text,
        nullable=False
    )

    start_time = Column(
        Float,
        nullable=False
    )

    end_time = Column(
        Float,
        nullable=False
    )

    # Relationship kwenda Ayah
    ayah = relationship(
        "Ayah",
        back_populates="segments"
    )
    words = relationship(
    "SegmentWord",
    back_populates="segment",
    cascade="all, delete-orphan"
)