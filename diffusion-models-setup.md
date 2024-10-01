# Diffusion Models Setup Guide

## Text2Img Inference
For the Text2Img custom pipeline, you need to download the ONNX branch of the Stable Diffusion model from:

https://huggingface.co/CompVis/stable-diffusion-v1-4

First, install Git LFS (Large File Storage) to manage model files:

```
git lfs install
```
We're only interested in the onnx branch of this repository, so use the following command to clone it:

```
git clone -b onnx https://huggingface.co/CompVis/stable-diffusion-v1-4

```

Once downloaded, update the path of your models in the Revit plugin to point to the locally downloaded models.


## Img2Img Inference with Stable Diffusion XL
For Img2Img inference, you will need the following models:

https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0

https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0

These models do not have an onnx branch, so we will export the PyTorch models to ONNX format and save them locally.

Navigate to the onnx-utilities directory of your project:

```
cd aectech-stable-diffusion/onnx-utilities
```
Activate the virtual environment, if not already activated.

Install the required Python libraries (if you haven't done so yet):

```
pip install -r requirements.txt
```
Use the following command to export the PyTorch model to ONNX format:
```
optimum-cli export onnx --model stabilityai/stable-diffusion-xl-base-1.0 --task image-to-image sd_xl_base_1_0/
```

For more details on the export process, refer to the official documentation:

https://huggingface.co/docs/optimum/main/en/exporters/onnx/usage_guides/export_a_model