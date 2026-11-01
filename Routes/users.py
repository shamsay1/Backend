from fastapi import APIRouter,Depends,HTTPException,status
from Schemas.users import UserCreate,UserResposne
from Models.database import get_db,AsyncSession
from Models import User
from sqlalchemy import func
from Services.password import password_hash
from typing import List
from sqlalchemy.future import select
router = APIRouter(prefix="/users",tags=['Users'])

@router.get("/all_users",response_model=List[UserResposne])
async def get_users(db: AsyncSession = Depends(get_db)):
    query = await db.execute(select(User))
    users = query.scalars().all()
    return users
@router.get("/count_users")
async def count_users(db: AsyncSession = Depends(get_db)):
    query = await db.execute(select(func.count()).select_from(User))
    all_users = query.scalar()
    return {"active_users" : all_users}

@router.post("/reg_users",response_model=UserResposne)
async def reg_users(request: UserCreate,db: AsyncSession = Depends(get_db)):
    query = await db.execute(select(User).filter(User.mobile==request.mobile))
    existing_user = query.scalars().first()
    if existing_user:
        raise HTTPException(status_code = status.HTTP_201_CREATED,details="User already created")
    hashed_password = password_hash(request.password)
    new_user = User(
        firstname=request.firstname,
        middlename=request.middlename,
        lastname=request.lastname,
        mobile=request.mobile,
        gender=request.gender,
        role=request.role,
        password=hashed_password,
    )
    db.add(new_user)
    await db.commit()
    return new_user

                    

