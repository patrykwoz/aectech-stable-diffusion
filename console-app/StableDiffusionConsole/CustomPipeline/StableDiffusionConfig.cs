using Microsoft.ML.OnnxRuntime;

namespace StableDiffusionConsole.CustomPipeline
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


            switch (ExecutionProviderTarget)
            {
                case ExecutionProvider.Cpu:
                    sessionOptions.AppendExecutionProvider_CPU();
                    return sessionOptions;
                default:
                case ExecutionProvider.Cuda:
                    // Adjust this based on your GPU memory
                    var gpuMemoryLimit = 11L * 1024 * 1024 * 1024;

                    var cudaOptionsDictionary = new Dictionary<string, string>();
                    cudaOptionsDictionary["device_id"] = DeviceId.ToString();
                    cudaOptionsDictionary["gpu_mem_limit"] = gpuMemoryLimit.ToString();
                    cudaOptionsDictionary["cudnn_conv_algo_search"] = "DEFAULT";
                    cudaOptionsDictionary["arena_extend_strategy"] = "kNextPowerOfTwo";
                    cudaOptionsDictionary["cudnn_conv_use_max_workspace"] = "1";
                    cudaOptionsDictionary["use_tf32"] = "0";

                    var cudaOptions = new OrtCUDAProviderOptions();
                    cudaOptions.UpdateOptions(cudaOptionsDictionary);


                    sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
                    sessionOptions.ExecutionMode = ExecutionMode.ORT_PARALLEL;
                    sessionOptions.EnableMemoryPattern = true;
                    sessionOptions.EnableProfiling = true;
                    sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);

                    return sessionOptions;
            }
        }
    }
}
