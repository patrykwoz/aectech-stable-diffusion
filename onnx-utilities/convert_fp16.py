import onnx
from onnxconverter_common import auto_mixed_precision, float16

# Load the ONNX model
# model_path = r"C:\Users\patry\source\repos\aectech-stable-diffusion\onnx_utilities\sd_v1_4_optimized\unet\model.onnx"
model_path =r"C:\Users\patry\source\repos\stable-diffusion-v1-4\unet\model_fp16.onnx"
model = onnx.load(model_path)

# Check the precision of the tensors
def check_precision(model):
    for initializer in model.graph.initializer:
        data_type = onnx.TensorProto.DataType.Name(initializer.data_type)
        print(f"Initializer: {initializer.name}, Data Type: {data_type}")

    for input in model.graph.input:
        data_type = onnx.TensorProto.DataType.Name(input.type.tensor_type.elem_type)
        print(f"Input: {input.name}, Data Type: {data_type}")

    for output in model.graph.output:
        data_type = onnx.TensorProto.DataType.Name(output.type.tensor_type.elem_type)
        print(f"Output: {output.name}, Data Type: {data_type}")

check_precision(model)

# Convert the model to FP16
# print("Converting the model to FP16...")

# print("Model converted to FP16 successfully!")
# onnx.save(model_fp16, r"C:\Users\patry\source\repos\stable-diffusion-v1-4\unet\model_fp16.onnx")