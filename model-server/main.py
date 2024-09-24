
import io
from typing import Union
from pathlib import Path
from PIL import Image

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from ml import obtain_image

app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/generate")
async def generate_image(
    prompt: str =  "a pixar style, highfidelity image of a building in the middle of a forest",
    file: UploadFile | None = File(None),
    guidance_scale: float = 7.5,
    strength: float = 0.5,
):
    if not file:
        raise ValueError("No file uploaded") 
    else:
        if file.content_type not in ["image/jpeg", "image/png"]:
            return {"error": "File format not supported. Please upload a .jpg or .png image."}

        image_data = await file.read()
        init_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        image = obtain_image(
            prompt,
            init_image=init_image,
            guidance_scale=guidance_scale,
            strength=strength,
        )
        image.save("image.png")
        return FileResponse("image.png")


@app.post("/generate-memory")
async def generate_image_memory(
    prompt: str =  "a pixar style, highfidelity image of a building in the middle of a forest",
    file: UploadFile | None = File(None),
    guidance_scale: float = 7.5,
    strength: float = 0.85,
):
    if not file:
        raise ValueError("No file uploaded")
    else:
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise ValueError("File format not supported. Please upload a .jpg or .png image.")

        image_data = await file.read()
        init_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        image = obtain_image(
            prompt,
            init_image=init_image,
            guidance_scale=guidance_scale,
            strength=strength,
        )

        memory_stream = io.BytesIO()
        image.save(memory_stream, format="PNG")
        memory_stream.seek(0)
        return StreamingResponse(memory_stream, media_type="image/png")