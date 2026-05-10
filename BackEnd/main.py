from typing import Annotated
from pathlib import Path

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware

import aiofiles
import os

from predict import predict

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = "uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.post("/predict")
async def predict_image(
    file: Annotated[UploadFile, File(...)]
):

    file_path = Path(UPLOAD_FOLDER) / file.filename

    async with aiofiles.open(file_path, "wb") as buffer:

        content = await file.read()

        await buffer.write(content)

    result = predict(str(file_path))

    return result

@app.get("/")
def home():
    return {"message": "Car Type Recognition API is running"}

from fastapi.responses import FileResponse


@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    return FileResponse("favicon.ico")