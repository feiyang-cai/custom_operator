import onnx
from onnx import helper

# Define the input names
input_name = "input"

# Define the output name
output_name = "output"

#custom_op_node = helper.make_node(
#    'CustomOpOne',  # The name of your custom operator
#    inputs=['input'],  # Inputs to the custom operator
#    outputs=['output'],  # Outputs of the custom operator
#    domain= 'test.customop',  # The custom operator domain you registered
#)

# Create input placeholders
input = helper.make_tensor_value_info(input_name, onnx.TensorProto.FLOAT, [5])

# Create output placeholder
output = helper.make_tensor_value_info(output_name, onnx.TensorProto.FLOAT, [5])

# Create custom operator node
custom_op_node = helper.make_node(
    op_type="TaxiNetDynamics",
    inputs=[input_name],
    outputs=[output_name],
    domain='custom_dynamics',
)

# Create graph
graph = helper.make_graph(
    nodes=[custom_op_node],
    name="custom_model",
    inputs=[input],
    outputs=[output],
)

# Create model
model = helper.make_model(graph, producer_name="Custom_Dynamics_Model")

# Save the model to an ONNX file
onnx_file_path = "custom_model.onnx"
onnx.save_model(model, onnx_file_path)