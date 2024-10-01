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


        }



        public void Dispose()
        {
            throw new NotImplementedException();
        }
    }
}
