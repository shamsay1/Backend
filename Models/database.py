from sqlalchemy.ext.asyncio import AsyncSession,create_async_engine,async_sessionmaker
from sqlalchemy.orm import sessionmaker,declarative_base

DATABASE_URL = "postgresql+asyncpg://postgres:shamsay321!@localhost:5432/quran_db"

engine = create_async_engine(DATABASE_URL,echo=False)

AsyncSessionLocal = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False
)
Base = declarative_base()

async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()


