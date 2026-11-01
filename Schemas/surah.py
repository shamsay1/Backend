from pydantic import BaseModel

class SurahResponse(BaseModel):
    id: int
    name_kiswahili: str
    name_arabic: str
    count_ayah: int
    class Config:
        from_attributes = True

