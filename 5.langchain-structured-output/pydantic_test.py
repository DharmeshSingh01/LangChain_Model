from pydantic import BaseModel, Field
from typing import Optional


class Student (BaseModel):
    name: str
    age: int
    height: Optional[float] = 5.6
    rank: int = Field(gt=0, lt=100)


new_student = {'name': 'Dharmesh', 'age': 30, 'rank': 6}

student = Student(**new_student)

print(student)
