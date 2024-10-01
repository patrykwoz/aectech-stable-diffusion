# Stable Diffusion Inference Inside Revit⚡

This repository provides a setup for running Stable Diffusion inference directly within Revit, offering three different approaches tailored to varying levels of expertise and deployment needs:

1. **Using a Python backend**:

    Use a Python server to perform Stable Diffusion inference and communicate with Revit through an API. This method provides quick access to a wide range of resources, tutorials, and the latest pre-trained models.
2. **ONNX Runtime with NuGet Package**:

   Integrate Stable Diffusion directly into Revit using the [OnnxStack.StableDiffusion](https://github.com/TensorStack-AI/OnnxStack/blob/master/OnnxStack.StableDiffusion/README.md) NuGet package. This method runs the model within the Revit environment using ONNX Runtime, providing a more seamless, local experience for users familiar with C#.
3. **ONNX Runtime with Custom Pipeline**:

    Build a fully custom Stable Diffusion pipeline directly inside Revit using ONNX Runtime. This setup requires a deeper understanding of both diffusion models and the ONNX Runtime framework. The pipeline can be customized based on your specific use case. Refer to the ONNX Runtime tutorial for guidance on setting up a C# pipeline.

These methods are ordered from the most straightforward and flexible to the more advanced setups that require greater technical knowledge and effort.

For individual users or small teams experimenting with Stable Diffusion, we recommend the Python backend, as it allows for quick experimentation and access to the latest model updates. This approach also supports a wide variety of tutorials and extensions, making it ideal for rapid prototyping and iteration.

Using ONNX Runtime inside Revit is more complex but offers significant advantages for deploying models at scale. This approach allows the model to run directly within the Revit environment, minimizing latency and improving performance for tasks that require real-time inference. Additionally, this method can be extended to edge computing, where model inference happens closer to the end user, reducing reliance on external servers.

Each method is designed to meet different deployment needs, from prototyping and development to production-scale deployments in enterprise environments.

## Getting started

### Initial Setup for All Approaches

To simplify development, we'll be working directly on Windows using Git Bash and PowerShell as our command-line interfaces.

1. Download and install [GitBash](https://git-scm.com/downloads)
2. Download and install the latest [Python](https://www.python.org/downloads/). Ensure that you add Python to your system's PATH during installation.
3. Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit). Make sure to match the CUDA version supported by your GPU.
4. Download and install [cuDNN](https://developer.nvidia.com/cudnn-downloads). After installation, ensure that cuDNN is set up correctly by copying the required files to the CUDA installation directory.
5. Install [PyTorch with CUDA support](https://pytorch.org/get-started/locally/). Run the following command, specifying the appropriate CUDA version (in this example, CUDA 12.4):

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```
6. Clone this repository to your local machine
```
git clone https://github.com/patrykwoz/aectech-stable-diffusion.git
```

### Setup for Developing the Python Server

1. Download and install [Visual Studio Code](https://code.visualstudio.com/).

2. Navigate to revit-server directory where the Python server code is located:
```
cd aectech-stable-diffusion/model-server
```
3. Setup python virtual environment
```
python -m venv venv
```
4. Activate the virtual environment
```
source venv/Scripts/activate
```

4. Install python libraries
```
pip install -r requirements.txt
```

5. Start the uvicorn server
```
uvicorn main:app --reload
```
The ```--reload``` flag is useful during development because it automatically reloads the server when you make code changes.

6. Verify that the server is running correctly by navigating to http://localhost:8000/docs. This link will show the automatically generated API documentation and allow you to test the available routes.

### Setup for developing the Revit plugin

Follow the steps in this [plugin development guide](./plugin-development.md) to create the Revit plugin from scratch. This guide provides additional details to help avoid potential pitfalls during setup.

1. Download and install [Visual Studio Community 2022](https://visualstudio.microsoft.com/), which will be used for building and debugging the Revit plugin.
2. Restore the .NET project dependencies by running the following command in the terminal:
```
dotnet restore
```
3. Rebuild the solution to ensure all dependencies are resolved and the project is correctly compiled:
```
dotnet build
```
4. Configure the path to Revit.exe as the startup executable within Visual Studio for debugging purposes:

 * In Visual Studio, right-click the project in Solution Explorer and go to Properties.
 * Under the Debug tab, specify the path to your local Revit installation's Revit.exe file. This ensures that Revit will be launched with the plugin when debugging.

 5. Start debugging by running the project in Visual Studio. This should launch Revit with the plugin loaded for development and testing.


