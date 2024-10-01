using Microsoft.ML.OnnxRuntime;
using OnnxStack.Core.Image;
using OnnxStack.StableDiffusion.Config;
using OnnxStack.StableDiffusion.Enums;
using OnnxStack.StableDiffusion.Pipelines;

namespace StableDiffusionConsole.OnnxStackPipeline
{
    public class InferenceApi : IDisposable
    {
        private bool _disposed = false;
        private readonly byte[] _model;
        private readonly List<string> _orderedInputNames;
        private readonly List<OrtValue> _inputData;
        private InferenceSession _inferenceSession;

        public InferenceApi()
        {
            var inputImagePath = @"C:\Users\patry\Desktop\512sample.png";

        }

        public async Task<string> RunInference()
        {
            var executionProvider = OnnxStack.Core.Config.ExecutionProvider.Cuda;

            var modelPath = @"C:\Users\patry\source\repos\aectech-stable-diffusion\onnx-utilities\sd_img_img_onnx_cuda";

            //var pipeline = StableDiffusionPipeline.CreatePipeline(modelPath);
            var pipeline = StableDiffusionXLPipeline.CreatePipeline(modelPath, ModelType.Base, 0, executionProvider, MemoryModeType.Minimum);
            //var pipeline = StableDiffusionPipeline.CreatePipeline(modelPath, ModelType.Base, 0, executionProvider, MemoryModeType.Maximum);


            var inputImagePath = @"C:\Users\patry\Desktop\512sample.png";
            var outputImagePath = @"C:\Users\patry\Desktop\Output_ImageToImage.png";
            
            var inputImage = await OnnxImage.FromFileAsync(inputImagePath);

            var promptOptions = new PromptOptions
            {
                DiffuserType = DiffuserType.ImageToImage,
                Prompt = "Photo of a building in a forest.",
                InputImage = inputImage

            };

            var schedulerOptions = pipeline.DefaultSchedulerOptions with
            {
                // How much the output should look like the input
                Strength = 0.8f,
                InferenceSteps = 10
            };

            var result = await pipeline.GenerateImageAsync(promptOptions, schedulerOptions);

            await result.SaveAsync(outputImagePath);

            await pipeline.UnloadAsync();

            return outputImagePath;

        }

        public void Dispose()
        {
            throw new NotImplementedException();
        }
    }
}
