import json
import os
import asyncio

from sqlalchemy import select

from Models.database import AsyncSessionLocal
from Models.ayah import Ayah


OUTPUT_FILE = "Datasets/faatiha/faatiha_dataset.json"


async def create_dataset():

    os.makedirs("Datasets/faatiha", exist_ok=True)

    async with AsyncSessionLocal() as session:

        # Faatiha = Surah number 1
        # Kwa database yako tunachukua Aya 7 za mwanzo.
        result = await session.execute(
            select(Ayah)
            .order_by(Ayah.id)
            .limit(7)
        )

        ayahs = result.scalars().all()

        dataset = []

        for ayah in ayahs:

            item = {
                "surah": 1,
                "surah_name": "Al-Faatiha",
                "ayah_id": ayah.id,
                "ayah_number": ayah.number,
                "text": ayah.text,
                "audio": f"ayah_{ayah.number:03d}.mp3"
            }

            dataset.append(item)

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            dataset,
            file,
            ensure_ascii=False,
            indent=2
        )

    print(f"\nDataset created successfully:")
    print(OUTPUT_FILE)

    print(f"\nTotal Ayahs: {len(dataset)}")


if __name__ == "__main__":
    asyncio.run(create_dataset())