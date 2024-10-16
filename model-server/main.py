
import io
from typing import Union
from pathlib import Path
from PIL import Image, ImageOps

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from ml import text_to_image, obtain_image, obtain_control_net_image

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
    
@app.post("/text-to-image")
async def generate_image_from_text(prompt: str = "Astronaut riding a horse"):
    image = text_to_image(prompt)
    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/png")

@app.post("/generate-memory")
async def generate_image_memory(
    prompt: str =  "a pixar style, highfidelity image of a building in the middle of a forest",
    file: UploadFile | None = File(None),
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    strength: float = 0.85,
    seed: int = None,
):
    if not file:
        raise ValueError("No file uploaded")
    else:
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise ValueError("File format not supported. Please upload a .jpg or .png image.")

        image_data = await file.read()
        init_image = Image.open(io.BytesIO(image_data)).convert("RGB")
        init_image = crop_and_resize_image(init_image)

        init_image.save("init_image.png")

        image = obtain_image(
            prompt,
            init_image=init_image,
            guidance_scale=guidance_scale,
            strength=strength,
            num_inference_steps=num_inference_steps,
            seed=seed,
        )

        memory_stream = io.BytesIO()
        image.save(memory_stream, format="PNG")
        memory_stream.seek(0)
        return StreamingResponse(memory_stream, media_type="image/png")

@app.post("/generate-controlnet")
async def generate_control_net_image(
    prompt: str = "aerial view, a futuristic house in a bright foggy jungle, hard lighting",
    negative_prompt: str = "low quality, bad quality, sketches",
    file: UploadFile = File(...),
    num_inference_steps: int = 50,
    guidance_scale: float = 5.0,
    controlnet_conditioning_scale: float = 0.5,
    seed: int = None,
):
    if not file:
        raise ValueError("No file uploaded")
    else:
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise ValueError("File format not supported. Please upload a .jpg or .png image.")

    image_data = await file.read()
    init_image = Image.open(io.BytesIO(image_data)).convert("RGB")
    init_image = crop_and_resize_image(init_image)

    image = obtain_control_net_image(
        prompt,
        negative_prompt,
        init_image=init_image,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        controlnet_conditioning_scale=controlnet_conditioning_scale,
        seed=seed,
    )

    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/png")

@app.post("/invert-rgb")
async def invert_rgb(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise ValueError("File format not supported. Please upload a .jpg or .png image.")

    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    image = crop_and_resize_image(image)
    image = ImageOps.invert(image)
    memory_stream = io.BytesIO()
    image.save(memory_stream, format="PNG")
    memory_stream.seek(0)
    return StreamingResponse(memory_stream, media_type="image/png")
    

def crop_and_resize_image(image: Image.Image, size: tuple[int, int] = (1024, 1024)) -> Image.Image:
    width, height = image.size
    if width > height:
        left = (width - height) / 2
        right = (width + height) / 2
        top = 0
        bottom = height
    else:
        left = 0
        right = width
        top = (height - width) / 2
        bottom = (height + width) / 2

    image = image.crop((left, top, right, bottom))
    image = image.resize(size)
    return image