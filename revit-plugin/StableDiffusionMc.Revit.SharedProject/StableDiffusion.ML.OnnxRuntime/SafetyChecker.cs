﻿using Microsoft.ML.OnnxRuntime.Tensors;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;

namespace StableDiffusionMc.Revit.StableDiffusion.ML.OnnxRuntime
{
    public static class SafetyChecker
    {

        private static DenseTensor<float> ReorderTensor(Tensor<float> inputTensor)
        {
            //reorder from batch channel height width to batch height width channel
            var inputImagesTensor = new DenseTensor<float>(new[] { 1, 224, 224, 3 });
            for (int y = 0; y < inputTensor.Dimensions[2]; y++)
            {
                for (int x = 0; x < inputTensor.Dimensions[3]; x++)
                {
                    inputImagesTensor[0, y, x, 0] = inputTensor[0, 0, y, x];
                    inputImagesTensor[0, y, x, 1] = inputTensor[0, 1, y, x];
                    inputImagesTensor[0, y, x, 2] = inputTensor[0, 2, y, x];
                }
            }

            return inputImagesTensor;
        }
        private static DenseTensor<float> ClipImageFeatureExtractor(Tensor<float> imageTensor, StableDiffusionConfig config)
        {
            // Create the image from tensor data
            var image = new Image<Rgba32>(config.Width, config.Height);

            // Process the pixel rows using ProcessPixelRows and PixelAccessor
            image.ProcessPixelRows(accessor =>
            {
                for (var y = 0; y < config.Height; y++)
                {
                    var pixelSpan = accessor.GetRowSpan(y);

                    for (var x = 0; x < config.Width; x++)
                    {
                        pixelSpan[x] = new Rgba32(
                            (byte)(Math.Round(Math.Clamp((imageTensor[0, 0, y, x] / 2 + 0.5), 0, 1) * 255)),
                            (byte)(Math.Round(Math.Clamp((imageTensor[0, 1, y, x] / 2 + 0.5), 0, 1) * 255)),
                            (byte)(Math.Round(Math.Clamp((imageTensor[0, 2, y, x] / 2 + 0.5), 0, 1) * 255))
                        );
                    }
                }
            });

            // Resize the image
            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(224, 224),
                    Mode = ResizeMode.Crop
                });
            });

            // Preprocess the image for the DenseTensor
            var input = new DenseTensor<float>(new[] { 1, 3, 224, 224 });
            var mean = new[] { 0.485f, 0.456f, 0.406f };
            var stddev = new[] { 0.229f, 0.224f, 0.225f };

            // Use ProcessPixelRows for preprocessing the resized image
            image.ProcessPixelRows(accessor =>
            {
                for (int y = 0; y < image.Height; y++)
                {
                    var pixelSpan = accessor.GetRowSpan(y);

                    for (int x = 0; x < image.Width; x++)
                    {
                        input[0, 0, y, x] = ((pixelSpan[x].R / 255f) - mean[0]) / stddev[0];
                        input[0, 1, y, x] = ((pixelSpan[x].G / 255f) - mean[1]) / stddev[1];
                        input[0, 2, y, x] = ((pixelSpan[x].B / 255f) - mean[2]) / stddev[2];
                    }
                }
            });

            return input;
        }

    }
}
