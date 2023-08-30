from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from slayer import Slayer
from markdown import markdown


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Replace with the origin of your React client
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResumeRequest(BaseModel):
    resume: str
    description: str
    title: str


@app.post("/process/")
async def process(request: ResumeRequest):
    slayer = Slayer(request.resume, request.description, request.title)
    md = await slayer.process()
    md = md.replace('\n', '')
    return {"data": markdown(md, output_format='html')}