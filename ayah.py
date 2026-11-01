import json
import asyncio

from Models.database import AsyncSessionLocal
from Models.surah import Surah
from Models.ayah import Ayah

from sqlalchemy import select


async def import_ayahs():

    async with AsyncSessionLocal() as db:

        try:

            with open(
                "quran-dataset/data/quran.json",
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

            for surah_data in data["surahs"]:

                # Tafuta Surah iliyopo kwenye database
                result = await db.execute(
                    select(Surah).where(
                        Surah.number == surah_data["number"]
                    )
                )

                surah = result.scalar_one_or_none()

                if not surah:
                    print(
                        f"Surah {surah_data['number']} "
                        f"haipo kwenye database"
                    )
                    continue

                # Import Aya za Surah hiyo
                for ayah_data in surah_data["ayahs"]:

                    ayah = Ayah(
                        number=ayah_data["number"],
                        text=ayah_data["text"],
                        juzuu=ayah_data["juz"],

                        # Relationship na Surah
                        surah=surah
                    )

                    db.add(ayah)

            await db.commit()

            print("Ayahs zimeingizwa successfully!")

        except Exception as e:

            await db.rollback()

            print("Error:", e)


if __name__ == "__main__":
    asyncio.run(import_ayahs())