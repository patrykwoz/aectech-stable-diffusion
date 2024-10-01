using SixLabors.ImageSharp;
using StableDiffusionMc.Revit.StableDiffusion.ML.OnnxRuntime;

namespace StableDiffusionConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Simple Stable Diffusion Inference Console Application");

            var prompt = "Astronaut riding a green horse on mars";
            // Update paths to your models
            var modelsBasePath = @"C:\Users\patry\source\repos\stable-diffusion-v1-4";

            var config = new StableDiffusionConfig
            {
                NumInferenceSteps = 15,
                GuidanceScale = 7.5,
                ExecutionProviderTarget = StableDiffusionConfig.ExecutionProvider.Cuda,
                DeviceId = 0,
                TextEncoderOnnxPath = Path.Combine(modelsBasePath, "text_encoder", "model.onnx"),
                UnetOnnxPath = Path.Combine(modelsBasePath, "unet", "model.onnx"),
                VaeDecoderOnnxPath = Path.Combine(modelsBasePath, "vae_decoder", "model.onnx"),
            };

            Console.WriteLine("Starting inference...");
            var image = UNet.Inference(prompt, config);

            if (image == null)
            {
                throw new Exception("Failed to generate image");
            }

            var generatedImageFileName = $"generated_{DateTime.Now:yyyyMMddHHmmss}.png";
            var genImgPath = Path.Combine(Path.GetTempPath(), generatedImageFileName);

            Console.WriteLine($"Saving generated image to: {genImgPath}");
            image.SaveAsPng(genImgPath);

            Console.WriteLine("Image saved!");
            Console.WriteLine(genImgPath);

            Console.WriteLine("Inference complete!");
        }
    }
}