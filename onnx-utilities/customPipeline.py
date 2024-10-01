import torch
import inspect
from diffusers import AutoencoderKL, UNet2DConditionModel, StableDiffusionXLPipeline
from diffusers.schedulers import EulerDiscreteScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from PIL import Image
from tqdm.auto import tqdm

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    """
    Rescale `noise_cfg` according to `guidance_rescale`.
    """
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

class CustomSDXLPipeline:
    def __init__(self, model_id="stabilityai/stable-diffusion-xl-base-1.0"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load SDXL components
        self.vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae").to(self.device)
        self.tokenizer_one = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
        self.tokenizer_two = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer_2")
        self.text_encoder_one = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder").to(self.device)
        self.text_encoder_two = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder_2").to(self.device)
        self.unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet").to(self.device)
        
        # Initialize scheduler
        self.scheduler = EulerDiscreteScheduler.from_pretrained(model_id, subfolder="scheduler")

    def encode_prompt(self, prompt, negative_prompt=""):
        # Tokenize and encode the prompt with both text encoders
        text_input_one = self.tokenizer_one(prompt, padding="max_length", max_length=self.tokenizer_one.model_max_length, truncation=True, return_tensors="pt")
        text_input_two = self.tokenizer_two(prompt, padding="max_length", max_length=self.tokenizer_two.model_max_length, truncation=True, return_tensors="pt")
        
        with torch.no_grad():
            text_embeddings_one = self.text_encoder_one(text_input_one.input_ids.to(self.device))[0]
            text_embeddings_two = self.text_encoder_two(text_input_two.input_ids.to(self.device))[0]
        
        # Concatenate the embeddings
        prompt_embeds = torch.cat([text_embeddings_one, text_embeddings_two], dim=-1)
        
        # Encode negative prompt if provided
        if negative_prompt:
            uncond_input_one = self.tokenizer_one(negative_prompt, padding="max_length", max_length=self.tokenizer_one.model_max_length, return_tensors="pt")
            uncond_input_two = self.tokenizer_two(negative_prompt, padding="max_length", max_length=self.tokenizer_two.model_max_length, return_tensors="pt")
            
            with torch.no_grad():
                uncond_embeddings_one = self.text_encoder_one(uncond_input_one.input_ids.to(self.device))[0]
                uncond_embeddings_two = self.text_encoder_two(uncond_input_two.input_ids.to(self.device))[0]
            
            uncond_embeds = torch.cat([uncond_embeddings_one, uncond_embeddings_two], dim=-1)
        else:
            uncond_embeds = torch.zeros_like(prompt_embeds)
        
        return prompt_embeds, uncond_embeds

    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    def generate(self, prompt, height=1024, width=1024, num_inference_steps=50, guidance_scale=7.5, negative_prompt=None, guidance_rescale=0.0):
        # Encode prompts
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(prompt, negative_prompt)
        
        # Prepare latents
        latents = torch.randn(
            (1, self.unet.config.in_channels, height // 8, width // 8),
            device=self.device
        )
        latents = latents * self.scheduler.init_noise_sigma
        
        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        
        # Prepare extra step kwargs
        extra_step_kwargs = self.prepare_extra_step_kwargs(None, None)
        
        # Create tensor for added time IDs
        add_time_ids = self._get_add_time_ids(height, width, dtype=prompt_embeds.dtype)
        
        # Create tensor for added text embeds (you may need to adjust this based on your specific implementation)
        add_text_embeds = torch.zeros_like(prompt_embeds[:, :1, :])
        
        # Denoising loop
        do_classifier_free_guidance = guidance_scale > 1.0
        for i, t in enumerate(tqdm(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
            
            # predict the noise residual
            added_cond_kwargs = {"text_embeds": add_text_embeds, "time_ids": add_time_ids}
            noise_pred = self.unet(
                latent_model_input,
                t,
                encoder_hidden_states=torch.cat([negative_prompt_embeds, prompt_embeds]) if do_classifier_free_guidance else prompt_embeds,
                cross_attention_kwargs=None,
                added_cond_kwargs=added_cond_kwargs,
                return_dict=False,
            )[0]
            
            # perform guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
            
            if do_classifier_free_guidance and guidance_rescale > 0.0:
                # Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
                noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=guidance_rescale)
            
            # compute the previous noisy sample x_t -> x_t-1
            latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
        
        # Decode latents to image
        latents = 1 / self.vae.config.scaling_factor * latents
        with torch.no_grad():
            image = self.vae.decode(latents).sample
        
        # Convert to PIL Image
        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
        image = (image * 255).round().astype("uint8")[0]
        image = Image.fromarray(image)
        
        return image

    def _get_add_time_ids(self, height, width, dtype):
        # This is a placeholder implementation. You may need to adjust this based on your specific requirements.
        return torch.tensor([height, width]).to(dtype=dtype, device=self.device).unsqueeze(0)

# Usage example
pipeline = CustomSDXLPipeline()
image = pipeline.generate(
    "A majestic lion sitting on a throne, digital art",
    negative_prompt="blurry, low quality",
    guidance_scale=7.5,
    guidance_rescale=0.7
)
image.save("sdxl_generated_image.png")