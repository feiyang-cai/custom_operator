import torch.nn as nn
import torch
import os
import onnx
from onnx import helper

p_normalizer = 6.366468343804353
theta_normalizer = 17.248858791583547

def create_gan(gan_model_path):
    class GANNet(nn.Module):
        def __init__(self):
            super(GANNet, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(4, 260), # 256 + 4 = 260, 256 original + 4 for input
                nn.ReLU(),

                nn.Linear(260, 260),
                nn.ReLU(),

                nn.Linear(260, 260),
                nn.ReLU(),

                nn.Linear(260, 260),
                nn.ReLU(),

                nn.Linear(260, 20),
                nn.ReLU(),

                nn.Linear(20, 12),
                nn.ReLU(),

                nn.Linear(12, 12),
                nn.ReLU(),

                nn.Linear(12, 6),
            )

        def forward(self, x):
            return self.main(x)
    
    gan = GANNet()
    params = torch.load(gan_model_path)
    for name, param in gan.main.named_parameters():
        original_param = params[name]
        if name == '0.weight':
            temp = torch.tensor([[1.0, 0.0, 0.0, 0.0],
                                 [0.0, 1.0, 0.0, 0.0],
                                 [0.0, 0.0, 1.0, 0.0],
                                 [0.0, 0.0, 0.0, 1.0]], dtype=torch.float32)
            param.data = torch.cat((original_param, temp), dim=0)
        elif name == '0.bias':
            param.data = torch.cat((original_param, torch.ones(4, dtype=torch.float32)*10.0), dim=0) # we move 0.8 to avoid the relu function
        elif name in ['2.weight', '4.weight', '6.weight']:
            temp = torch.zeros((256, 4), dtype=torch.float32)
            original_param = torch.cat((original_param, temp), dim=1)
            temp = torch.zeros((4, 260), dtype=torch.float32)
            temp[0, 256] = 1.0
            temp[1, 257] = 1.0
            temp[2, 258] = 1.0
            temp[3, 259] = 1.0
            param.data = torch.cat((original_param, temp), dim=0)
        elif name == '8.weight':
            temp = torch.zeros((16, 4), dtype=torch.float32)
            original_param = torch.cat((original_param, temp), dim=1)
            temp = torch.zeros((4, 260), dtype=torch.float32)
            temp[0, 256] = 1.0
            temp[1, 257] = 1.0
            temp[2, 258] = 1.0
            temp[3, 259] = 1.0
            param.data = torch.cat((original_param, temp), dim=0)
        elif name == '10.weight':
            temp = torch.zeros((8, 4), dtype=torch.float32)
            original_param = torch.cat((original_param, temp), dim=1)
            temp = torch.zeros((4, 20), dtype=torch.float32)
            temp[0, 16] = 1.0
            temp[1, 17] = 1.0
            temp[2, 18] = 1.0
            temp[3, 19] = 1.0
            param.data = torch.cat((original_param, temp), dim=0)
        elif name == '12.weight':
            temp = torch.zeros((8, 4), dtype=torch.float32)
            original_param = torch.cat((original_param, temp), dim=1)
            temp = torch.zeros((4, 12), dtype=torch.float32)
            temp[0, 8] = 1.0
            temp[1, 9] = 1.0
            temp[2, 10] = 1.0
            temp[3, 11] = 1.0
            param.data = torch.cat((original_param, temp), dim=0)
        elif name in ['2.bias', '4.bias', '6.bias', '8.bias', '10.bias', '12.bias']:
            param.data = torch.cat((original_param, torch.zeros(4, dtype=torch.float32)), dim=0)
        elif name == '14.weight':
            temp = torch.zeros((2, 4), dtype=torch.float32)
            original_param = torch.cat((original_param, temp), dim=1)
            temp = torch.zeros((4, 12), dtype=torch.float32)
            temp[0, 8] = 1.0
            temp[1, 9] = 1.0
            temp[2, 10] = 1.0
            temp[3, 11] = 1.0
            param.data = torch.cat((temp, original_param), dim=0)
        elif name == '14.bias':
            param.data = torch.cat((torch.ones(4, dtype=torch.float32)*(-10.0), original_param), dim=0) # we move 0.8 to avoid the relu function
        
    return gan

def create_norm_net():
    class NormNet(nn.Module):
        def __init__(self):
            super(NormNet, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(4, 4, bias=False),
            )

        def forward(self, x):
            return self.main(x) 

    norm_net = NormNet()
    for param in norm_net.main.parameters():
        param.data = torch.FloatTensor([[1.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0/p_normalizer, 0.0],
                                        [0.0, 0.0, 0.0, 1.0/theta_normalizer]])

    return norm_net

def create_controller_net():
    class ControllerNet(nn.Module):
        def __init__(self):
            super(ControllerNet, self).__init__()
            self.main = nn.Linear(6, 5, bias=False)

        def forward(self, x):
            return self.main(x)
    
    controller_net = ControllerNet()
    for param in controller_net.main.parameters():
        #param.data = torch.FloatTensor([[-0.74, -0.44]]) # (1, 2)
        param.data = torch.FloatTensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                                       [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                                       [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                                       [0.0, 0.0, 0.0, 0.0, -0.74, -0.44]]) 

    return controller_net

def create_extract_net():
    class ExtractNet(nn.Module):
        def __init__(self):
            super(ExtractNet, self).__init__()
            self.main = nn.Linear(5, 5, bias=False)

        def forward(self, x):
            return self.main(x)
    
    extract_net = ExtractNet()
    for param in extract_net.main.parameters():
        param.data = torch.FloatTensor([[1.0, 0.0, 0.0, 0.0, 0.0], # z0
                                        [0.0, 1.0, 0.0, 0.0, 0.0], # z1
                                        [0.0, 0.0, p_normalizer, 0.0, 0.0], # p
                                        [0.0, 0.0, 0.0, theta_normalizer/180.0*torch.pi, 0.0], # theta
                                        [0.0, 0.0, 0.0, 0.0, 1.0/180.0*torch.pi]])

    return extract_net

def create_pre_dynamics_net(norm_net, gan_net, controller_net, extract_net):
    class PreDynamicsNet(nn.Module):
        def __init__(self, norm_net, gan_net, controller_net, extract_net):
            super(PreDynamicsNet, self).__init__()
            self.gan_net = gan_net
            self.norm_net = norm_net
            self.controller_net = controller_net
            self.extract_net = extract_net

        def forward(self, x):
            x = self.norm_net(x)
            x = self.gan_net(x)
            x = self.controller_net(x)
            x = self.extract_net(x)
            return x
    
    pre_dynamics_net = PreDynamicsNet(norm_net, gan_net, controller_net, extract_net)
    return pre_dynamics_net

def create_post_dynamics_net():
    class PostDynamicsNet(nn.Module):
        def __init__(self):
            super(PostDynamicsNet, self).__init__()
            self.main = nn.Linear(5, 4, bias=False)

        def forward(self, x):
            return self.main(x)
    
    post_dynamics_net = PostDynamicsNet()
    for param in post_dynamics_net.main.parameters():
        param.data = torch.FloatTensor([[1.0, 0.0, 0.0, 0.0, 0.0],
                                        [0.0, 1.0, 0.0, 0.0, 0.0],
                                        [0.0, 0.0, 1.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0, 1.0/torch.pi*180.0, 0.0]]) # (4, 5)

    return post_dynamics_net

if __name__ == '__main__':
    gan_model_path = "./models/full_mlp_supervised.pth"
    pre_dynamics_model_path = "./models/pre_dynamics.onnx"
    post_dynamics_model_path = "./models/post_dynamics.onnx"

    step = 1
    substep = 20

    assert os.path.exists(gan_model_path) or \
        (os.path.exists(pre_dynamics_model_path) and \
            os.path.exists(post_dynamics_model_path))
    
    if not (os.path.exists(pre_dynamics_model_path) and os.path.exists(post_dynamics_model_path)):
        # create these models first
        
        # pre dynamics model        
        norm_net = create_norm_net()
        gan_net = create_gan(gan_model_path)
        controller_net = create_controller_net()
        extract_net = create_extract_net()
        pre_dynamics_model = create_pre_dynamics_net(norm_net, gan_net, controller_net, extract_net)

        # post dynamics model
        denorm_net = create_post_dynamics_net()

        # export them to onnx
        x = torch.randn(1, 4)
        print(pre_dynamics_model(x).shape)
        torch.onnx.export(pre_dynamics_model, 
                          x, 
                          pre_dynamics_model_path, 
                          input_names=['x'],
                          output_names=['x_pre_dynamics'],
                          verbose=True)

        x = torch.randn(1, 5)
        torch.onnx.export(denorm_net,
                          x,
                          post_dynamics_model_path,
                          input_names=['x_post_dynamics'],
                          output_names=['x_'],
                          verbose=True)

    
    # load them
    pre_dynamics_model = onnx.load(pre_dynamics_model_path)
    post_dynamics_model = onnx.load(post_dynamics_model_path)


    # create the step dynamics from substep dynamics node
    step_dynamics_nodes = []
    for i in range(substep):
        node = helper.make_node(
            op_type="TaxiNetDynamics",
            inputs=[f"x_substep_{i}" if i > 0 else "x_pre_dynamics"],
            outputs=[f"x_substep_{i+1}" if i < substep-1 else "x_post_dynamics"],
            name=f"dynamics_substep_{i}",
            domain='custom_dynamics',
        )
        step_dynamics_nodes.append(node)
    
    input = helper.make_tensor_value_info("x_pre_dynamics", onnx.TensorProto.FLOAT, [5])
    output = helper.make_tensor_value_info("x_post_dynamics", onnx.TensorProto.FLOAT, [5])
    step_dynamics_graph = helper.make_graph(
        nodes=step_dynamics_nodes,
        name="step_dynamics",
        inputs =[input],
        outputs=[output]
    )

    step_dynamics_model = helper.make_model(step_dynamics_graph, producer_name='Feiyang Cai')
    
    onnx.save_model(step_dynamics_model, "./models/step_dynamics.onnx")

    # create the step model
    step_model = onnx.ModelProto()
    step_model.CopyFrom(pre_dynamics_model)
    step_model.graph.node.extend(step_dynamics_model.graph.node)
    step_model.graph.node.extend(post_dynamics_model.graph.node)
    step_model.graph.initializer.extend(post_dynamics_model.graph.initializer)
    # revise the name of the node to avoid conflict // TODO: find a better way to do this
    step_model.graph.node[-1].name = "MatMul_18"
    step_model.graph.output.pop()
    step_model.graph.output.extend(post_dynamics_model.graph.output)

    onnx.save_model(step_model, "./models/step_model.onnx")


    system_nodes = []
    system_initializers = []
    for i in range(step):
        step_graph = onnx.GraphProto()
        step_graph.CopyFrom(step_model.graph)
        for step_node in step_graph.node:
            step_node.name += f"_step_{i}"
            for input_i, _ in enumerate(step_node.input):
                if step_node.input[input_i] == step_graph.input[0].name:
                    step_node.input[input_i] = f"x_step_{i}"
                else:
                    input_name = step_node.input[input_i]
                    new_name = input_name + f"_step_{i}"
                    step_node.input[input_i] += f"_step_{i}"
                    for initializer in step_graph.initializer:
                        if initializer.name == input_name:
                            initializer.name = new_name
            for output_i, _ in enumerate(step_node.output):
                if step_node.output[output_i] == step_graph.output[0].name:
                    step_node.output[output_i] = f"x_step_{i+1}"
                else:
                    output_name = step_node.output[output_i]
                    new_name = output_name + f"_step_{i}"
                    step_node.output[output_i] += f"_step_{i}"
                    for initializer in step_graph.initializer:
                        if initializer.name == output_name:
                            initializer.name = new_name
                    
        system_nodes.extend(step_graph.node)
        system_initializers.extend(step_graph.initializer)

    input = helper.make_tensor_value_info("x_step_0", onnx.TensorProto.FLOAT, [1, 4])
    output = helper.make_tensor_value_info(f"x_step_{step}", onnx.TensorProto.FLOAT, [1, 4])
    system_graph = helper.make_graph(
        nodes=system_nodes,
        initializer=system_initializers,
        name="system",
        inputs =[input],
        outputs=[output]
    )

    system_model = helper.make_model(system_graph, producer_name='Feiyang Cai')
    print(system_model.graph.output)
    onnx.save_model(system_model, "./models/system_model.onnx")