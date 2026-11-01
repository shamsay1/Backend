from pydantic import BaseModel, ConfigDict
from typing import Optional


class AyahResponse(BaseModel):
    id: int
    surah_id: int
    number: Optional[int] = None
    text: Optional[str] = None
    juzuu: Optional[int] = None

    model_config = ConfigDict(
        from_attributes=True
    )