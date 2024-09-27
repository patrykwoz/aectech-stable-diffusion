# import torch
# from diffusers import StableDiffusionXLImg2ImgPipeline
import onnx

print("Checking the model...")
# onnx_model = onnx.load(r"C:\Users\patry\source\repos\aectech-stable-diffusion\revit-plugin\models\unet\model.onnx")

model_path = r"C:\Users\patry\source\repos\aectech-stable-diffusion\onnx_utilities\sd_img_img_onnx\unet\model.onnx"
# onnx_model = onnx.load(model_path)
vae_encoder_model_path = r"C:\Users\patry\source\repos\aectech-stable-diffusion\onnx_utilities\sd_img_img_onnx\vae_encoder\model.onnx"
onnx.checker.check_model(model_path)
print("UNET ONNX model exported and checked successfully!")
onnx.checker.check_model(vae_encoder_model_path)
print("VAE Encoder ONNX model exported and checked successfully!")
