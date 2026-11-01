from typing import List
from Models.database import AsyncSessionLocal
from fastapi import APIRouter, Depends,HTTPException
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from Models.ayah import Ayah
from Models.database import get_db
from Schemas.ayah import AyahResponse
from Models.ayah_segment import AyahSegment
from Models.segment_word import SegmentWord


router = APIRouter(
    prefix="/ayah",
    tags=["Ayah"]
)


@router.get(
    "/surah/{surah_id}",
    response_model=List[AyahResponse]
)
async def get_ayahs_by_surah(
    surah_id: int,
    db: AsyncSession = Depends(get_db)
):
    query = await db.execute(
        select(Ayah)
        .where(
            Ayah.surah_id == surah_id
        )
        .order_by(
            Ayah.number.asc()
        )
    )

    ayahs = query.scalars().all()

    return ayahs
# ============================================================
# GET AYAH SEGMENTS
# ============================================================

@router.get("/ayah/{ayah_id}/segments")
async def get_ayah_segments(ayah_id: int):

    async with AsyncSessionLocal() as db:

        result = await db.execute(
            select(AyahSegment)
            .where(AyahSegment.ayah_id == ayah_id)
            .order_by(AyahSegment.segment_number)
        )

        segments = result.scalars().all()

        if not segments:
            raise HTTPException(
                status_code=404,
                detail=f"Hakuna segments za Ayah ID {ayah_id}."
            )

        response = []

        for segment in segments:

            word_result = await db.execute(
                select(SegmentWord)
                .where(
                    SegmentWord.segment_id == segment.id
                )
                .order_by(SegmentWord.word_number)
            )

            words = word_result.scalars().all()

            response.append({
                "id": segment.id,
                "ayah_id": segment.ayah_id,
                "segment_number": segment.segment_number,
                "text": segment.text,
                "start_time": segment.start_time,
                "end_time": segment.end_time,

                "words": [
                    {
                        "id": word.id,
                        "word_number": word.word_number,
                        "text": word.text,
                        "start_time": word.start_time,
                        "end_time": word.end_time,
                    }
                    for word in words
                ],
            })

        return {
            "success": True,
            "ayah_id": ayah_id,
            "segments": response,
        }