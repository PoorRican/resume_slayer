from fastapi import FastAPI
from pydantic import BaseModel
from slayer import Slayer


app = FastAPI()


class ResumeRequest(BaseModel):
    resume: str
    description: str
    title: str


@app.post("/process/")
async def process(request: ResumeRequest):
    slayer = Slayer(request.resume, request.description, request.title)
    return {"data": await slayer.process()}
