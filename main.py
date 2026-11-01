
from fastapi import FastAPI
from Models import User
from sqlalchemy.future import select
from Models.database import engine,Base,get_db,AsyncSession
from Services.password import password_hash
from Models import Surah
from Routes import users
from Routes import ayah
from Models import Ayah
from Models import AyahSegment
from Models import SegmentWord
from quran_learn import router as learn_router
from Routes import surah
from quran_recitation import router as quran_router

app = FastAPI()
app.include_router(users.router)
app.include_router(surah.router)
app.include_router(ayah.router)
app.include_router(quran_router)
app.include_router(learn_router)

@app.on_event("startup")
async def startup_event():
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    async for db in get_db():
        await seed_data(db)
async def seed_data(db: AsyncSession):
    password="12345678"
    query = await db.execute(select(User).filter(User.mobile=="0634263868"))
    existing_user = query.scalars().first()
    if not existing_user:
        admin = User(
            firstname="SHAMIS",
            middlename="NASSOR",
            lastname="ALI",
            mobile="0634263868",
            gender="Male",
            role="admin",
            password= password_hash(password)
            
            
        )
        db.add(admin)
        await db.commit()
print("=== ROUTES ZILIZOSAJILIWA ===")
for r in app.routes:
    print(type(r).__name__, getattr(r, "path", ""))
print("=============================")
        

        


