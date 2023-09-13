from fastapi import BackgroundTasks, FastAPI, WebSocket
from starlette.websockets import WebSocketState
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
from slayer import Slayer
from time import sleep
from uuid import uuid4
from supabase import create_client, Client
from typing import Union

url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_KEY")
supabase: Client = create_client(url, key)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000",
                   '*-poorrican.vercel.app'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ResumeRequest(BaseModel):
    resume: str
    description: str
    title: str


async def process_resume_request(job_id: str, request: ResumeRequest) -> None:
    """ Upload data to supabase and process request.

    Supabase is used as an intermediary to store processed resume. User provided data is immediately inserted
    into the database. `Slayer.process()` is executed as a background task, then `Job` row is updated.

    :param job_id: UUID for job id. This is provided back to the frontend.
    :param request: Data from user request

    :return: None
    """
    # Perform your time-intensive process here
    # Update the database with the results
    # You can use any database library or ORM of your choice
    resume_id = str(uuid4())
    supabase \
        .table('Resumes') \
        .insert({"id": resume_id, "text": request.resume}) \
        .execute()
    supabase \
        .table('Jobs') \
        .insert({"id": job_id, "title": request.title, "description": request.description, "resume": resume_id}) \
        .execute()

    slayer = Slayer(request.resume, request.description, request.title)
    md = await slayer.process()

    supabase \
        .table('Jobs') \
        .update({"processed": md}) \
        .eq("id", job_id) \
        .execute()


@app.post("/process/")
async def process(request: ResumeRequest, background_tasks: BackgroundTasks):
    """ Process and improve resume data supplied via POST

    A job UUID is immediately provided to listen to events and to retrieve processed data. The data is processed in the
    background, therefore, the frontend application should listen for events via supabase.

    :param request: POST data
    :param background_tasks: argument to enable background tasks

    :return: a job id is returned
    """
    job_id = str(uuid4())  # Generate a unique job ID
    background_tasks.add_task(process_resume_request, job_id, request)
    return job_id


@app.websocket("/ws/{test_option}")
async def process_websocket(websocket: WebSocket, test_option: str = None):
    """ Process incoming data """

    test = False
    crash = False

    if test_option:
        if test_option == "bypass":
            test = True
        elif test_option == "crash":
            crash = True
        else:
            raise ValueError("Incorrect GET arguments for `test_option`")

    await websocket.accept()

    while websocket.application_state == WebSocketState.CONNECTED:
        # Receive data from the WebSocket connection
        resume = await websocket.receive_text()
        title = await websocket.receive_text()
        description = await websocket.receive_text()
        # generate unique id's
        job_id = str(uuid4())  # Generate a unique job ID
        resume_id = str(uuid4())  # Generate a unique job ID

        md = ""

        if not (test or crash):
            supabase \
                .table('Resumes') \
                .insert({"id": resume_id, "text": resume}) \
                .execute()
            supabase \
                .table('Jobs') \
                .insert({"id": job_id, "title": title, "description": description, "resume": resume_id}) \
                .execute()

            slayer = Slayer(resume, description, title)
            md = await slayer.process()

            supabase \
                .table('Jobs') \
                .update({"processed": md}) \
                .eq("id", job_id) \
                .execute()

        elif test:
            # allow for test to detect progress component
            sleep(1)
            md = "Correct websocket sequence received"

        elif crash:
            sleep(1)
            raise BrokenPipeError("Simulated Crash")

        # Send a response back to the WebSocket connection
        await websocket.send_text(md)

        await websocket.close()


@app.get('/test')
def test_page():
    return "Test Successful"
