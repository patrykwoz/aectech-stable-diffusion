// See https://aka.ms/new-console-template for more information

using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
using SixLabors.ImageSharp;
using StableDiffusionMc.Revit.StableDiffusion.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Linq;


namespace StableDiffusionConsole
{
    class Program
    {
        static void Main(string[] args)
        {
            Console.WriteLine("Hello World!");

            //var inferenceApi = new InferenceApi();

            //inferenceApi.CreateInferenceSession();

            var prompt = "Astronaut riding a green horse on mars";

            var tokens = TokenizeText2(prompt);

            //var inferenceApi = new InferenceApi();
            //NDArray input = new NDArray(new float[] { 1, 2, 3, 4, 5 });
            //var prediction = inferenceApi.Predict(input);
            //var tokenizedText = InferenceApi.TokenizeText("Hello World!");
            //var preprocessedText = InferenceApi.PreprocessText("Astronatu on mars riding a green horse");

            var modelsBasePath = @"C:\Users\patry\source\repos\stable-diffusion-v1-4";
            var unetModelBasPath = @"C:\Users\patry\source\repos\aectech-stable-diffusion\onnx_utilities\sd_v1_4_optimized";
            var config = new StableDiffusionConfig
            {
                // Number of denoising steps
                NumInferenceSteps = 15,
                // Scale for classifier-free guidance
                GuidanceScale = 7.5,
                // Set your preferred Execution Provider. Currently (GPU, DirectML, CPU) are supported in this project.
                // ONNX Runtime supports many more than this. Learn more here: https://onnxruntime.ai/docs/execution-providers/
                // The config is defaulted to CUDA. You can override it here if needed.
                // To use DirectML EP intall the Microsoft.ML.OnnxRuntime.DirectML and uninstall Microsoft.ML.OnnxRuntime.GPU
                ExecutionProviderTarget = StableDiffusionConfig.ExecutionProvider.Cuda,
                // Set GPU Device ID.
                DeviceId = 0,
                // Update paths to your models
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

        public static int[] TokenizeText2 (string text)
        {
            var tokenizerPath = @"C:\Users\patry\source\repos\StableDiffusionConsole\models\cliptokenizer.onnx";
            //tokenizerPath = @"C:\Users\patry\source\repos\aectech-stable-diffusion\onnx_utilities\sd_img_img_onnx_cuda\text_encoder\model.onnx";
            //tokenizerPath = @"C:\Users\patry\source\repos\aectech-stable-diffusion\onnx_utilities\tokenizer_onnx\model.onnx";
            var sessionOptions = new SessionOptions();
            sessionOptions.RegisterOrtExtensions();
            sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            sessionOptions.AppendExecutionProvider_CUDA(0);

            var tokenizeSession = new InferenceSession(tokenizerPath, sessionOptions);

            Console.WriteLine("Model loaded successfully!");
            Console.WriteLine("Model input metadata:");

            foreach (var input in tokenizeSession.InputMetadata)
            {
                Console.WriteLine($"Input Name: {input.Key}");
                var nodeInfo = input.Value;
                Console.WriteLine($"  Element Type: {nodeInfo.ElementType}");
                Console.WriteLine($"  Dimensions: {string.Join(", ", nodeInfo.Dimensions.Select(d => d.ToString()))}");
            }

            Console.WriteLine("Model output metadata:");
            foreach (var output in tokenizeSession.OutputMetadata)
            {
                Console.WriteLine($"Output Name: {output.Key}");
                var nodeInfo = output.Value;
                Console.WriteLine($"  Element Type: {nodeInfo.ElementType}");
                Console.WriteLine($"  Dimensions: {string.Join(", ", nodeInfo.Dimensions.Select(d => d.ToString()))}");
            }

            using var inputTensor = OrtValue.CreateTensorWithEmptyStrings(OrtAllocator.DefaultInstance, new long[] { 1 });
            inputTensor.StringTensorSetElementAt(text.AsSpan(), 0);

            var inputs = new Dictionary<string, OrtValue>
            {
                { "string_input", inputTensor }
            };

            //var inputTensor = new DenseTensor<string>(new string[] { text }, new int[] { 1 });
            //var inputString = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<string>("string_input", inputTensor) };

            using var runOptions = new RunOptions();
            using var tokens = tokenizeSession.Run(
                runOptions,
                inputs,
                tokenizeSession.OutputNames);

            var inputIds = tokens[0].GetTensorDataAsSpan<long>();

            var InputIdsInt = new int[inputIds.Length];
            for (int i = 0; i < inputIds.Length; i++)
            {
                InputIdsInt[i] = (int)inputIds[i];
            }

            Console.WriteLine(String.Join(" ", InputIdsInt));

            var modelMaxLength = 77;
            // Pad array with 49407 until length is modelMaxLength
            if (InputIdsInt.Length < modelMaxLength)
            {
                var pad = Enumerable.Repeat(49407, 77 - InputIdsInt.Length).ToArray();
                InputIdsInt = InputIdsInt.Concat(pad).ToArray();
            }
            return InputIdsInt;
        }
    }
}