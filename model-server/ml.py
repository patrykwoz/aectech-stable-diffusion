import torch
from diffusers import StableDiffusionXLImg2ImgPipeline
from diffusers.utils import load_image
from PIL import Image

# model_id = "stabilityai/stable-diffusion-xl-base-1.0"
model_id = "stabilityai/stable-diffusion-xl-refiner-1.0"

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
# attempting to speed up the model
# pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
# pipe.enable_model_cpu_offload()
pipe = pipe.to("cuda")

def obtain_image(
    prompt: str,
    init_image: Image,
    num_inference_steps: int = 50,
    strength: float = 0.85,
    guidance_scale: float = 7.5,
) -> Image:
    # generator to make generation deterministic/reproducible
    # generator = None if seed is None else torch.Generator("cuda").manual_seed(seed)
    print(f"Using device: {pipe.device}")
    print(f"Guidance scale: {guidance_scale}")
    print(f"Strength: {strength}")
    image: Image = pipe(
        prompt,
        image=init_image,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        
    ).images[0]
    return image