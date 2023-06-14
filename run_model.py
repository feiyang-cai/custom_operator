import onnxruntime as rt
import numpy as np
# Load the ONNX model containing the custom operator

shared_library = "./lib/libcustom_dynamics.so"
so = rt.SessionOptions()
so.register_custom_ops_library(shared_library)

model_path = 'custom_model.onnx'
sess1 = rt.InferenceSession(model_path, so)

input_name_0 = sess1.get_inputs()[0].name
output_name = sess1.get_outputs()[0].name

input_shape0 = sess1.get_inputs()[0].shape
output_shape = sess1.get_outputs()[0].shape


#input_0 = np.ones((3, )).astype(np.float32)
#input_1 = np.zeros((3, )).astype(np.float32)
input_0 = np.random.random(input_shape0).astype(np.float32)

res = sess1.run([output_name], {input_name_0: input_0})
print(input_0, res)