from .database import Base
from sqlalchemy import Column,String,Integer,Enum

class User(Base):
    __tablename__ = "users"
    id = Column(Integer,primary_key=True,index=True)
    firstname = Column(String(255),nullable=False)
    middlename = Column(String(255),nullable=False)
    lastname = Column(String(255),nullable=False)
    mobile = Column(String(255),nullable=False)
    gender = Column(Enum('Male','Female',name="gender_enum"),nullable=False)
    role = Column(Enum('customer','admin','sub_admin',name="role_enum"))
    password = Column(String(255),nullable=False)
    status = Column(String(255),default="Active")
