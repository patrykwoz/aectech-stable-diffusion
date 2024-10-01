using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using NumSharp;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using StableDiffusion.ML.OnnxRuntime;
using StableDiffusionMc.Revit.StableDiffusion.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using static System.Net.Mime.MediaTypeNames;

namespace StableDiffusionConsole
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
            
        }

        public NDArray Predict(NDArray input)
        {
            
            

            return null;
        }

        public void CreateInferenceSession()
        {
            // Create session options for custom op of extensions
            var sessionOptions = new SessionOptions();
            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
            //var cudaOptions = new OrtCUDAProviderOptions();
            //sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
            sessionOptions.AppendExecutionProvider_CUDA(0);

            var prompt = new string[] { "a photograph of an astronaut riding a horse" };
            var height = 512;
            var width = 512;
            var numInferenceSteps = 15;
            var guidanceScale = 7.5;
            var batchSize = prompt.Length;

            var modelsBasePath = @"C:\Users\patry\source\repos\aectech-stable-diffusion\onnx_utilities\sd_img_img_onnx_cuda";

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

            var tokenizeSession = new InferenceSession(config.TokenizerOnnxPath, sessionOptions);
            var inputTensor = new DenseTensor<string>(new string[] { prompt[0] }, new int[] { 1 });
            var inputString = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<string>("string_input", inputTensor) };
            // Run session and send the input data in to get inference output. 
            var tokens = tokenizeSession.Run(inputString);





        }



        //public static int[] TokenizeText(string prompt)
        //{
        //    // Create session options for custom op of extensions
        //    var sessionOptions = new SessionOptions();
        //    sessionOptions.RegisterOrtExtensions();

        //    var tokenizerPath = @"C:\Users\patry\source\repos\StableDiffusionConsole\models\cliptokenizer.onnx";

        //    // Create an InferenceSession from the onnx clip tokenizer.
        //    var tokenizeSession = new InferenceSession(tokenizerPath, sessionOptions);
        //    var inputTensor = new DenseTensor<string>(new string[] { prompt }, new int[] { 1 });
        //    var inputString = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<string>("string_input", inputTensor) };
        //    // Run session and send the input data in to get inference output. 
        //    var tokens = tokenizeSession.Run(inputString);


        //    var inputIds = (tokens.ToList().First().Value as IEnumerable<long>).ToArray();
        //    Console.WriteLine(String.Join(" ", inputIds));

        //    // Cast inputIds to Int32
        //    var InputIdsInt = inputIds.Select(x => (int)x).ToArray();

        //    var modelMaxLength = 77;
        //    // Pad array with 49407 until length is modelMaxLength
        //    if (InputIdsInt.Length < modelMaxLength)
        //    {
        //        var pad = Enumerable.Repeat(49407, 77 - InputIdsInt.Length).ToArray();
        //        InputIdsInt = InputIdsInt.Concat(pad).ToArray();
        //    }

        //    return InputIdsInt;

        //}

        //public static DenseTensor<float> PreprocessText(string prompt)
        //{
        //    // Load the tokenizer and text encoder to tokenize and encode the text.
        //    var textTokenized = TokenizeText(prompt);
        //    var textPromptEmbeddings = TextEncoder(textTokenized).ToArray();

        //    // Create uncond_input of blank tokens
        //    var uncondInputTokens = CreateUncondInput();
        //    var uncondEmbedding = TextEncoder(uncondInputTokens).ToArray();

        //    // Concant textEmeddings and uncondEmbedding
        //    DenseTensor<float> textEmbeddings = new DenseTensor<float>(new[] { 2, 77, 768 });

        //    for (var i = 0; i < textPromptEmbeddings.Length; i++)
        //    {
        //        textEmbeddings[0, i / 768, i % 768] = uncondEmbedding[i];
        //        textEmbeddings[1, i / 768, i % 768] = textPromptEmbeddings[i];
        //    }
        //    return textEmbeddings;
        //}

        //public static int[] CreateUncondInput()
        //{
        //    // Create an array of empty tokens for the unconditional input.
        //    var blankTokenValue = 49407;
        //    var modelMaxLength = 77;
        //    var inputIds = new List<Int32>();
        //    inputIds.Add(49406);
        //    var pad = Enumerable.Repeat(blankTokenValue, modelMaxLength - inputIds.Count()).ToArray();
        //    inputIds.AddRange(pad);

        //    return inputIds.ToArray();
        //}

        //public static DenseTensor<float> TextEncoder(int[] tokenizedInput)
        //{
        //    // Create input tensor.
            
        //    var input_ids = TensorHelper.CreateTensor(tokenizedInput, new[] { 1, tokenizedInput.Count() });

        //    var input = new List<NamedOnnxValue> { NamedOnnxValue.CreateFromTensor<int>("input_ids", input_ids) };

        //    // Set CUDA EP
        //    var sessionOptions = new SessionOptions();
        //    sessionOptions.RegisterOrtExtensions();
        //    OrtCUDAProviderOptions cudaOptions = new OrtCUDAProviderOptions();
        //    sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);

        //    var textEncoderPath = @"C:\Users\patry\source\repos\StableDiffusionConsole\models\text_encoder\model.onnx";
        //    var encodeSession = new InferenceSession(textEncoderPath, sessionOptions);
        //    // Run inference.
        //    var encoded = encodeSession.Run(input);

        //    var lastHiddenState = (encoded.ToList().First().Value as IEnumerable<float>).ToArray();
        //    var lastHiddenStateTensor = TensorHelper.CreateTensor(lastHiddenState.ToArray(), new[] { 1, 77, 768 });

        //    return lastHiddenStateTensor;

        //}



        public void Dispose()
        {
            throw new NotImplementedException();
        }
    }
}
