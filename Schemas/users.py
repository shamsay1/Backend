from pydantic import BaseModel

class UserCreate(BaseModel):
    firstname:str
    middlename: str
    lastname: str
    mobile: str
    gender: str
    role: str
    password: str

class UserResposne(BaseModel):
    id: int
    firstname:str
    middlename: str
    lastname: str
    mobile: str
    gender: str
    role: str
    
