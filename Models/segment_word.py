from sqlalchemy import (
    Column,
    Integer,
    Text,
    Float,
    ForeignKey
)
from sqlalchemy.orm import relationship

from .database import Base


class SegmentWord(Base):

    __tablename__ = "segment_words"

    # =====================================================
    # PRIMARY KEY
    # =====================================================

    id = Column(
        Integer,
        primary_key=True,
        index=True
    )

    # =====================================================
    # FOREIGN KEY
    # =====================================================

    segment_id = Column(
        Integer,
        ForeignKey(
            "ayah_segments.id",
            ondelete="CASCADE"
        ),
        nullable=False,
        index=True
    )

    # =====================================================
    # WORD NUMBER
    # =====================================================

    word_number = Column(
        Integer,
        nullable=False
    )

    # =====================================================
    # ARABIC WORD
    # =====================================================

    text = Column(
        Text,
        nullable=False
    )

    # =====================================================
    # AUDIO TIMESTAMPS
    # =====================================================

    start_time = Column(
        Float,
        nullable=False
    )

    end_time = Column(
        Float,
        nullable=False
    )

    segment = relationship(
        "AyahSegment",
        back_populates="words"
    )