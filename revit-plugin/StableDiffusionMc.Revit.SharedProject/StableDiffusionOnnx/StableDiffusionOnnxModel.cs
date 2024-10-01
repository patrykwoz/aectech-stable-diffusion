using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Image;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Pipelines;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using StableDiffusionMc.Revit.StableDiffusion.ML.OnnxRuntime;
using System.Diagnostics;
using System.IO;

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
            var modelsBasePath = @"C:\Users\patry\source\repos\stable-diffusion-v1-4";
            var config = new StableDiffusion.ML.OnnxRuntime.StableDiffusionConfig
            {
                NumInferenceSteps = 15,
                GuidanceScale = guidanceScale,
                ExecutionProviderTarget = StableDiffusion.ML.OnnxRuntime.StableDiffusionConfig.ExecutionProvider.Cuda,
                DeviceId = 0,
                TextEncoderOnnxPath = Path.Combine(modelsBasePath, "text_encoder", "model.onnx"),
                UnetOnnxPath = Path.Combine(modelsBasePath, "unet", "model.onnx"),
                VaeDecoderOnnxPath = Path.Combine(modelsBasePath, "vae_decoder", "model.onnx"),
            };

            Debug.WriteLine($"Inference with prompt: {prompt}");

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

        public static async Task<string> InferWithOnnxStack(
            string imagePath,
            string prompt,
            double guidanceScale = 7.5,
            float strength = 0.85f
            )
        {

            var executionProvider = OnnxStack.Core.Config.ExecutionProvider.Cuda;

            var modelPath = @"C:\Users\patry\source\repos\aectech-stable-diffusion\onnx-utilities\sd_img_img_onnx_cuda";

            var pipeline = StableDiffusionXLPipeline.CreatePipeline(modelPath, ModelType.Base, 0, executionProvider, MemoryModeType.Minimum);

            var outputImagePath = @"C:\Users\patry\Desktop\Output_ImageToImage.png";

            var croppedImage = CropAndResizeImage(Image.Load<Rgba32>(imagePath), 1024, 1024);
            croppedImage.Save(imagePath);
            
            var inputImage = await OnnxImage.FromFileAsync(imagePath);
            
            var promptOptions = new PromptOptions
            {
                DiffuserType = DiffuserType.ImageToImage,
                Prompt = prompt,
                InputImage = inputImage

            };

            var schedulerOptions = pipeline.DefaultSchedulerOptions with
            {
                // How much the output should look like the input
                Strength = strength,
                InferenceSteps = 10
            };

            Debug.WriteLine($"Inference with prompt: {prompt}");

            var result = await pipeline.GenerateImageAsync(promptOptions, schedulerOptions);

            Debug.WriteLine($"Finished Inference");

            await result.SaveAsync(outputImagePath);

            await pipeline.UnloadAsync();

            return outputImagePath;
        }

        public static Image<Rgba32> CropAndResizeImage(Image<Rgba32> image, int width = 1024, int height = 1024)
        {
            int originalWidth = image.Width;
            int originalHeight = image.Height;
            Rectangle cropRectangle;

            if (originalWidth > originalHeight)
            {
                int left = (originalWidth - originalHeight) / 2;
                cropRectangle = new Rectangle(left, 0, originalHeight, originalHeight);
            }
            else
            {
                int top = (originalHeight - originalWidth) / 2;
                cropRectangle = new Rectangle(0, top, originalWidth, originalWidth);
            }

            image.Mutate(x => x.Crop(cropRectangle));
            image.Mutate(x => x.Resize(width, height));

            return image;
        }
    }
}
