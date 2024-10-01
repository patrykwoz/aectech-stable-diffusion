using SixLabors.ImageSharp;
using StableDiffusionConsole.CustomPipeline;
using StableDiffusionConsole.OnnxStackPipeline;

namespace StableDiffusionConsole
{
    class Program
    {
        static async Task Main(string[] args)
        {
            Console.WriteLine("Simple Stable Diffusion Inference Console Application");

            Console.WriteLine("Inference using OnnxStack (Image to Image) pipeline...");
            Console.WriteLine("Starting inference...");
            var inferenceApi = new InferenceApi();
            var imagePath = await inferenceApi.RunInference();

            Console.WriteLine($"Inference complete! Image saved to: {imagePath}");

            var prompt = "Astronaut riding a green horse on mars";
            var modelsBasePath = @"C:\Users\patry\source\repos\stable-diffusion-v1-4";

            var config = new StableDiffusionConfig
            {
                NumInferenceSteps = 50,
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