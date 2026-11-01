from sqlalchemy import Column, Integer, String, ForeignKey
from sqlalchemy.orm import relationship
from Models.database import Base


class Ayah(Base):
    __tablename__ = "ayah"

    id = Column(Integer, primary_key=True, index=True)

    surah_id = Column(
        Integer,
        ForeignKey("surahs.id"),
        nullable=False
    )

    number = Column(Integer)
    text = Column(String(1500))
    juzuu = Column(Integer)

    surah = relationship(
        "Surah",
        back_populates="ayahs"
    )
    segments = relationship(
    "AyahSegment",
    back_populates="ayah",
    cascade="all, delete-orphan"
)