from .database import AsyncSession,Base
from sqlalchemy import Integer,Column,String
from sqlalchemy.orm import relationship
class Surah(Base):
    __tablename__ = "surahs"
    id = Column(Integer,primary_key=True,index=True)
    number = Column(Integer)
    name_kiswahili = Column(String(100))
    name_arabic = Column(String(100))
    count_ayah = Column(Integer)
    juzuu = Column(Integer)
    ayahs = relationship(
        "Ayah",
        back_populates="surah",
        cascade="all, delete-orphan"
    )


    
