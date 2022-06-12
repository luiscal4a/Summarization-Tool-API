from typing import Optional
from fastapi import FastAPI
from summarization import *
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


class Document(BaseModel):
    content: str
    model: str
    min_length: int = 75
    max_length: int = 300

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/display/")
async def display_text(document: Document):
    return displayText(document.content)

@app.post("/summarize/")
async def summarize_text(document: Document):
    return summarize(document.content, document.model, document.min_length, document.max_length)
