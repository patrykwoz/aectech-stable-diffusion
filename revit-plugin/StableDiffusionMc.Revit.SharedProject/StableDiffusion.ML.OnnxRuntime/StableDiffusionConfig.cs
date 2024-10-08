﻿using Microsoft.ML.OnnxRuntime;

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
        // default props
        public int NumInferenceSteps = 15;
        public ExecutionProvider ExecutionProviderTarget = ExecutionProvider.Cuda;
        public double GuidanceScale = 7.5;
        public int Height = 512;
        public int Width = 512;
        public int DeviceId = 0;


        public string TokenizerOnnxPath = @"C:\Users\patry\source\repos\aectech-stable-diffusion\revit-plugin\models\cliptokenizer.onnx";
        public string TextEncoderOnnxPath = "";
        public string UnetOnnxPath = "";
        public string VaeDecoderOnnxPath = "";

        // default directory for images
        public string ImageOutputPath = "";

        public SessionOptions GetSessionOptionsForEp()
        {
            var sessionOptions = new SessionOptions();


            switch (this.ExecutionProviderTarget)
            {
                case ExecutionProvider.DirectML:
                    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    sessionOptions.EnableMemoryPattern = false;
                    sessionOptions.AppendExecutionProvider_DML(this.DeviceId);
                    sessionOptions.AppendExecutionProvider_CPU();
                    return sessionOptions;
                case ExecutionProvider.Cpu:
                    sessionOptions.AppendExecutionProvider_CPU();
                    return sessionOptions;
                default:
                case ExecutionProvider.Cuda:
                    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    //default to CUDA, fall back on CPU if CUDA is not available.
                    sessionOptions.AppendExecutionProvider_CUDA(this.DeviceId);
                    sessionOptions.AppendExecutionProvider_CPU();
                    //sessionOptions = SessionOptions.MakeSessionOptionWithCudaProvider(cudaProviderOptions);
                    return sessionOptions;

            }

        }
    }
}
