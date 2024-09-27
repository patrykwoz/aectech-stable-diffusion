using System;
using System.IO;
using System.Collections.Generic;
using System.Text;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.Memory;
using StableDiffusionMc.Revit.StableDiffusion.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime;

namespace StableDiffusionMc.Revit.StableDiffusionOnnx
{
    public static class StableDiffusionOnnxModel
    {
        public static async Task<string> InferWithOnnx(
            string imagePath,
            string prompt,
            double guidanceScale = 7.5,
            double strength = 0.85
            )
        {
            var modelsBasePath = @"C:\Users\patry\source\repos\aectech-stable-diffusion\revit-plugin\models";
            var config = new StableDiffusionConfig
            {
                // Number of denoising steps
                NumInferenceSteps = 15,
                // Scale for classifier-free guidance
                GuidanceScale = guidanceScale,
                // Set your preferred Execution Provider. Currently (GPU, DirectML, CPU) are supported in this project.
                // ONNX Runtime supports many more than this. Learn more here: https://onnxruntime.ai/docs/execution-providers/
                // The config is defaulted to CUDA. You can override it here if needed.
                // To use DirectML EP intall the Microsoft.ML.OnnxRuntime.DirectML and uninstall Microsoft.ML.OnnxRuntime.GPU
                ExecutionProviderTarget = StableDiffusionConfig.ExecutionProvider.Cuda,
                // Set GPU Device ID.
                DeviceId = 2,
                // Update paths to your models
                TextEncoderOnnxPath = Path.Combine(modelsBasePath, "text_encoder", "model.onnx"),
                UnetOnnxPath = Path.Combine(modelsBasePath, "unet", "model.onnx"),
                VaeDecoderOnnxPath = Path.Combine(modelsBasePath, "vae_decoder", "model.onnx"),
            };

            var image = UNet.Inference(prompt, config);

            if (image == null)
            {
                throw new Exception("Failed to generate image");
            }

            var generatedImageFileName = $"generated_{DateTime.Now:yyyyMMddHHmmss}.png";
            var genImgPath = Path.Combine(Path.GetTempPath(), generatedImageFileName);

            image.SaveAsPng(genImgPath);


            return genImgPath;
        }

        public static async Task<string> BasicInference (
            string imagePath,
            string prompt,
            double guidanceScale = 7.5,
            double strength = 0.85
            )
        {
            var modelsBasePath = @"C:\Users\patry\source\repos\aectech-stable-diffusion\revit-plugin\models";
            using var options = new SessionOptions();
            options.RegisterOrtExtensions();
            var session = new InferenceSession(modelsBasePath, options);

            return null;
        }
    }
}
