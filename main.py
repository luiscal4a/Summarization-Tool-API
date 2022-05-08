from typing import Optional
from fastapi import FastAPI
from summarization import *
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware


class Document(BaseModel):
    content: str

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
    print("hello")
    return displayText(document.content)

@app.post("/summarize/")
async def summarize_text(document: Document):
    print("hello")
    return summarize(document.content)


@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}