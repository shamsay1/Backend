from fastapi import APIRouter,Depends,HTTPException,Query
from sqlalchemy.future import select
from sqlalchemy import or_
from typing import List
from Models import Surah
from Models.database import AsyncSession,get_db
from Schemas.surah import SurahResponse
router = APIRouter(prefix="/surah",tags=["Surah"])

@router.get("/all_surah",response_model=List[SurahResponse])
async def all_surah(db: AsyncSession = Depends(get_db)):
    query = await db.execute(select(Surah))
    all_surah = query.scalars().all()
    return all_surah

@router.get(
    "/search_surah",
    response_model=List[SurahResponse]
)
async def search_surah(
    keyword: str = Query(
        ...,
        min_length=1,
        description="Search Surah by Arabic or Swahili name"
    ),
    db: AsyncSession = Depends(get_db)
):

    search_keyword = f"%{keyword.strip()}%"

    query = await db.execute(
        select(Surah)
        .where(
            or_(
                Surah.name_arabic.ilike(search_keyword),
                Surah.name_kiswahili.ilike(search_keyword)
            )
        )
        .order_by(Surah.id.asc())
    )

    surahs = query.scalars().all()

    return surahs


# =========================================================
# GET ONE SURAH
# =========================================================

@router.get(
    "/one_surah/{surah_id}",
    response_model=SurahResponse
)
async def one_surah(
    surah_id: int,
    db: AsyncSession = Depends(get_db)
):

    query = await db.execute(
        select(Surah)
        .where(Surah.id == surah_id)
    )

    surah = query.scalar_one_or_none()

    if surah is None:
        raise HTTPException(
            status_code=404,
            detail="Surah haikupatikana"
        )

    return surah