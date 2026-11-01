import json
import asyncio

from Models.database import AsyncSessionLocal
from Models.surah import Surah


async def import_surahs():

    async with AsyncSessionLocal() as db:

        try:

            with open(
                "quran-dataset/data/quran.json",
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

            for item in data["surahs"]:

                # Chukua juzuu ya Aya ya kwanza
                juzuu = item["ayahs"][0]["juz"]

                surah = Surah(
                    number=item["number"],
                    name_kiswahili=item["name_transliteration"],
                    name_arabic=item["name_arabic"],
                    count_ayah=item["counts"]["ayahs"],
                    juzuu=juzuu
                )

                db.add(surah)

            await db.commit()

            print("Surahs zimeingizwa successfully!")

        except Exception as e:

            await db.rollback()

            print("Error:", e)


if __name__ == "__main__":
    asyncio.run(import_surahs())