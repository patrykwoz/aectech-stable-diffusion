from transformers import CLIPTokenizer, pipeline, CLIPTextModel
import torch
from diffusers import LMSDiscreteScheduler, AutoencoderKL, UNet2DConditionModel
from tqdm.auto import tqdm

scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)



prompt = ["a photograph of an astronaut riding a horse"]

height = 1024                        # default height of Stable Diffusion
width = 1024                         # default width of Stable Diffusion

num_inference_steps = 10           # Number of denoising steps

guidance_scale = 7.5                # Scale for classifier-free guidance

generator = torch.manual_seed(0)    # Seed generator to create the inital latent noise

batch_size = len(prompt)

print("Loading the models...")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
unet = UNet2DConditionModel.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="unet")

torch_device = "cuda"
# vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device) 

print("Unet inputs and outputs: ")
print(unet.config)

max_length = tokenizer.model_max_length
print(max_length)

print("Tokenizing the prompt...")

text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")


print("Encoding the prompt...")
text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]
print("Text embeddings shape: ", text_embeddings.shape)

max_length = text_input.input_ids.shape[-1]
print("Max length after encoding: ", max_length)

uncond_input = tokenizer(
    [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
)
print("Unconditional input shape: ", uncond_input.input_ids.shape)

uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0] 
print("Unconditional embeddings shape: ", uncond_embeddings.shape)

text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
print("Text embeddings shape after concatenation: ", text_embeddings.shape)


latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8),
    generator=generator,
)
latents = latents.to(torch_device)
print("Latents shape after random initialization: ", latents.shape)

latents = latents * scheduler.init_noise_sigma
print("Latents shape after multiplying by init noise sigma: ", latents.shape)

scheduler.set_timesteps(num_inference_steps)

for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    print("Latent model input shape: ", latent_model_input.shape)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)
    print("Latent model input shape after scaling: ", latent_model_input.shape)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample