from fastapi import Depends, FastAPI
from pydantic import BaseModel,Field
from pydantic import BaseModel, Field, model_validator
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = [
    "https://your-frontend-domain.com",
    "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins, 
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"], 
)
@app.get("/")
async def read_root():
    return {"message": "Welcome To DDI Extraction it is Working Fine !"}

class TextInput(BaseModel):
    text: str

@app.post("/extract-ddis")
async def extract_ddis(input: TextInput):
    print("text :",input)
    return input.text