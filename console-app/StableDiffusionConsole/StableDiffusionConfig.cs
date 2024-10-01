using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.Text;

namespace StableDiffusionMc.Revit.StableDiffusion.ML.OnnxRuntime
{
    public class StableDiffusionConfig
    {
        public enum ExecutionProvider
        {
            DirectML = 0,
            Cuda = 1,
            Cpu = 2
        }

        public int NumInferenceSteps = 15;
        public ExecutionProvider ExecutionProviderTarget = ExecutionProvider.Cuda;
        public double GuidanceScale = 7.5;
        public int Height = 512;
        public int Width = 512;
        public int DeviceId = 0;

        public string TokenizerOnnxPath = @"C:\Users\patry\source\repos\StableDiffusionConsole\models\cliptokenizer.onnx";
        public string TextEncoderOnnxPath = "";
        public string UnetOnnxPath = "";
        public string VaeDecoderOnnxPath = "";

        // default directory for images
        public string ImageOutputPath = "";

        public SessionOptions GetSessionOptionsForEp()
        {
            var sessionOptions = new SessionOptions();
            sessionOptions.RegisterOrtExtensions();
            //sessionOptions.LogSeverityLevel = OrtLoggingLevel.ORT_LOGGING_LEVEL_VERBOSE;


            switch (this.ExecutionProviderTarget)
            {
                case ExecutionProvider.Cpu:
                    sessionOptions.AppendExecutionProvider_CPU();
                    return sessionOptions;
                default:
                case ExecutionProvider.Cuda:
                    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    //var cudaOptions = new OrtCUDAProviderOptions();
                    //sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
                    sessionOptions.AppendExecutionProvider_CUDA(this.DeviceId);
                    return sessionOptions;
            }
        }
    }
}
