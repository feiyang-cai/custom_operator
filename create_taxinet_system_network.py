import torch.nn as nn
import torch
import os
import onnx
from onnx import helper

p_normalizer = 6.366468343804353
theta_normalizer = 17.248858791583547

coeff_p = -0.2
coeff_theta = -0.1

def create_gan(gan_model_path, step):
    class GANNet(nn.Module):
        def __init__(self):
            super(GANNet, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(2+step*2, 258+(step-1)*2), # 256 + 4 = 260, 256 original + 4 for input
                nn.ReLU(),

                nn.Linear(258+(step-1)*2, 258+(step-1)*2),
                nn.ReLU(),

                nn.Linear(258+(step-1)*2, 258+(step-1)*2),
                nn.ReLU(),

                nn.Linear(258+(step-1)*2, 258+(step-1)*2),
                nn.ReLU(),

                nn.Linear(258+(step-1)*2, 18+(step-1)*2),
                nn.ReLU(),

                nn.Linear(18+(step-1)*2, 10+(step-1)*2),
                nn.ReLU(),

                nn.Linear(10+(step-1)*2, 10+(step-1)*2),
                nn.ReLU(),

                nn.Linear(10+(step-1)*2, 4+(step-1)*2),
            )

        def forward(self, x):
            return self.main(x)
    
    gan = GANNet()
    params = torch.load(gan_model_path)
    for name, param in gan.main.named_parameters():
        original_param = params[name]
        if name == '0.weight':
            temp = torch.zeros((256, (step-1)*2), dtype=torch.float32)
            param.data = torch.cat((original_param, temp), dim=1)

            block1 = torch.zeros((min(2, (step-1)*2), 2+step*2), dtype=torch.float32)
            block2 = torch.zeros((2, 2+step*2), dtype=torch.float32)
            block2[0, 2] = 1.0
            block2[1, 3] = 1.0

            if step > 1:
                block1.data[0, 4] = 1.0
                block1.data[1, 5] = 1.0

            param.data = torch.cat((param.data, block1, block2), dim=0)

            if step > 2:
                block3 = torch.zeros(((step-2)*2, 2+step*2), dtype=torch.float32)
                param.data = torch.cat((param.data, block3), dim=0)
                for i in range(1, (step-2)*2+1):
                    param.data[-i, -i] = 1.0



        elif name == '0.bias':
            param.data = torch.cat((original_param, torch.ones(2+(step-1)*2, dtype=torch.float32)*10.0), dim=0) # we move 0.8 to avoid the relu function

        elif name in ['2.weight', '4.weight', '6.weight']:
            temp = torch.zeros((256, 2+(step-1)*2), dtype=torch.float32)
            original_param = torch.cat((original_param, temp), dim=1)
            temp = torch.zeros((2+(step-1)*2, 258+(step-1)*2), dtype=torch.float32)
            temp[-1, -1] = 1.0
            temp[-2, -2] = 1.0
            if step > 1:
                temp[-3, -3] = 1.0
                temp[-4, -4] = 1.0
            param.data = torch.cat((original_param, temp), dim=0)
            
        elif name == '8.weight':
            temp = torch.zeros((16, 2+(step-1)*2), dtype=torch.float32)
            original_param = torch.cat((original_param, temp), dim=1)
            temp = torch.zeros((2+(step-1)*2, 258+(step-1)*2), dtype=torch.float32)
            temp[-1, -1] = 1.0
            temp[-2, -2] = 1.0
            if step> 1:
                temp[-3, -3] = 1.0
                temp[-4, -4] = 1.0
            param.data = torch.cat((original_param, temp), dim=0)

        elif name == '10.weight':
            temp = torch.zeros((8, 2+(step-1)*2), dtype=torch.float32)
            original_param = torch.cat((original_param, temp), dim=1)
            temp = torch.zeros((2+(step-1)*2, 18+(step-1)*2), dtype=torch.float32)
            temp[-1, -1] = 1.0
            temp[-2, -2] = 1.0
            if step>1:
                temp[-3, -3] = 1.0
                temp[-4, -4] = 1.0
            param.data = torch.cat((original_param, temp), dim=0)

        elif name == '12.weight':
            temp = torch.zeros((8, 2+(step-1)*2), dtype=torch.float32)
            original_param = torch.cat((original_param, temp), dim=1)
            temp = torch.zeros((2+(step-1)*2, 10+(step-1)*2), dtype=torch.float32)
            temp[-1, -1] = 1.0
            temp[-2, -2] = 1.0
            if step>1:
                temp[-3, -3] = 1.0
                temp[-4, -4] = 1.0
            param.data = torch.cat((original_param, temp), dim=0)

        elif name in ['2.bias', '4.bias', '6.bias', '8.bias', '10.bias', '12.bias']:
            param.data = torch.cat((original_param, torch.zeros(2+(step-1)*2, dtype=torch.float32)), dim=0)

        elif name == '14.weight':
            temp = torch.zeros((2, 2+(step-1)*2), dtype=torch.float32)
            original_param = torch.cat((original_param, temp), dim=1)
            temp = torch.zeros((2+(step-1)*2, 10+(step-1)*2), dtype=torch.float32)
            temp[-1, -1] = 1.0
            temp[-2, -2] = 1.0
            if step>1:
                temp[-3, -3] = 1.0
                temp[-4, -4] = 1.0
            param.data = torch.cat((temp, original_param), dim=0)

        elif name == '14.bias':
            param.data = torch.cat((torch.ones(2+(step-1)*2, dtype=torch.float32)*(-10.0), original_param), dim=0) # we move 0.8 to avoid the relu function
        
    return gan

def create_norm_net(step):
    class NormNet(nn.Module):
        def __init__(self):
            super(NormNet, self).__init__()
            self.main = nn.Sequential(
                nn.Linear(2+step*2, 2+step*2, bias=False),
            )

        def forward(self, x):
            return self.main(x) 

    norm_net = NormNet()
    for param in norm_net.main.parameters():
        
        param.data = torch.eye(2+step*2, dtype=torch.float32)
        param.data[2, 2] = 1.0/p_normalizer
        param.data[3, 3] = 1.0/theta_normalizer
        

    return norm_net

def create_controller_net(step):
    class ControllerNet(nn.Module):
        def __init__(self):
            super(ControllerNet, self).__init__()
            self.main = nn.Linear(4+(step-1)*2, 3+(step-1)*2, bias=False)
            #self.main = nn.Linear(6, 5, bias=False)

        def forward(self, x):
            return self.main(x)
    
    controller_net = ControllerNet()
    for param in controller_net.main.parameters():
        #param.data = torch.FloatTensor([[-0.74, -0.44]]) # (1, 2)
        temp = torch.eye(2+(step-1)*2, dtype=torch.float32)
        param.data = torch.cat((temp, torch.zeros((2+(step-1)*2, 2), dtype=torch.float32)), dim=1)
        param.data = torch.cat((param.data, torch.zeros((1, 2+step*2), dtype=torch.float32)), dim=0)
        param.data[-1, -2] = coeff_p
        param.data[-1, -1] = coeff_theta
        
        #param.data = torch.FloatTensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        #                               [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        #                               [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        #                               [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
        #                               [0.0, 0.0, 0.0, 0.0, -0.74, -0.44]]) 

    return controller_net

def create_extract_net(step):
    class ExtractNet(nn.Module):
        def __init__(self):
            super(ExtractNet, self).__init__()
            self.main = nn.Linear(3+(step-1)*2, 3+(step-1)*2, bias=False)

        def forward(self, x):
            return self.main(x)
    
    extract_net = ExtractNet()
    for param in extract_net.main.parameters():
        param.data = torch.eye(3+(step-1)*2, dtype=torch.float32)
        param.data[-1, -1] = 1.0/180.0*torch.pi
        param.data[min((step-1)*2, 2), min((step-1)*2, 2)] = p_normalizer
        param.data[min((step-1)*2, 2) + 1, min((step-1)*2, 2) + 1] = theta_normalizer/180.0*torch.pi
        
        
        #param.data = torch.FloatTensor([[1.0, 0.0, 0.0, 0.0, 0.0], # z0
        #                                [0.0, 1.0, 0.0, 0.0, 0.0], # z1
        #                                [0.0, 0.0, p_normalizer, 0.0, 0.0], # p
        #                                [0.0, 0.0, 0.0, theta_normalizer/180.0*torch.pi, 0.0], # theta
        #                                [0.0, 0.0, 0.0, 0.0, 1.0/180.0*torch.pi]])

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

def create_post_dynamics_net(step):
    class PostDynamicsNet(nn.Module):
        def __init__(self):
            super(PostDynamicsNet, self).__init__()
            self.main = nn.Linear(3+(step-1)*2, 2+(step-1)*2, bias=False)

        def forward(self, x):
            return self.main(x)
    
    post_dynamics_net = PostDynamicsNet()
    for param in post_dynamics_net.main.parameters():
        param.data = torch.eye(2+(step-1)*2, dtype=torch.float32)
        param.data[min(2*(step-1), 2)+1, min(2*(step-1), 2)+1] = 1.0/torch.pi*180.0
        param.data = torch.cat((param.data, torch.zeros((2+(step-1)*2, 1), dtype=torch.float32)), dim=1)
        
        #param.data = torch.FloatTensor([[1.0, 0.0, 0.0, 0.0, 0.0],
        #                                [0.0, 1.0, 0.0, 0.0, 0.0],
        #                                [0.0, 0.0, 1.0, 0.0, 0.0],
        #                                [0.0, 0.0, 0.0, 1.0/torch.pi*180.0, 0.0]]) # (4, 5)

    return post_dynamics_net

if __name__ == '__main__':
    gan_model_path = "./models/full_mlp_supervised.pth"

    step = 1
    system_nodes = []
    system_initializers = []
    for i in range(step, 0, -1):
        # pre dynamics model        
        norm_net = create_norm_net(step=i)
        gan_net = create_gan(gan_model_path, step=i)
        controller_net = create_controller_net(step=i)
        extract_net = create_extract_net(step=i)
        pre_dynamics_model = create_pre_dynamics_net(norm_net, gan_net, controller_net, extract_net)

        # post dynamics model
        denorm_net = create_post_dynamics_net(step=i)

        pre_dynamics_model_path = f"./models/pre_dynamics_step_{i}_{coeff_p}_{coeff_theta}.onnx"
        post_dynamics_model_path = f"./models/post_dynamics_step_{i}_{coeff_p}_{coeff_theta}.onnx"

        # export them to onnx
        x = torch.randn(1, 2+i*2)
        torch.onnx.export(pre_dynamics_model, 
                          x, 
                          pre_dynamics_model_path, 
                          input_names=['x'],
                          output_names=['x_pre_dynamics'],
                          verbose=False)

        x = torch.randn(1, 3+(i-1)*2)
        torch.onnx.export(denorm_net,
                          x,
                          post_dynamics_model_path,
                          input_names=['x_post_dynamics'],
                          output_names=['x_'],
                          verbose=False)
    
        # load them
        pre_dynamics_model = onnx.load(pre_dynamics_model_path)
        post_dynamics_model = onnx.load(post_dynamics_model_path)

        # create the step dynamics from substep dynamics node
        step_dynamics_nodes = []
        node = helper.make_node(
            op_type="TaxiNetDynamics",
            inputs=["x_pre_dynamics"],
            outputs=["x_post_dynamics"],
            name="dynamics",
            domain='custom_dynamics',
        )
        step_dynamics_nodes.append(node)
    
        input = helper.make_tensor_value_info("x_pre_dynamics", onnx.TensorProto.FLOAT, [3+(i-1)*2])
        output = helper.make_tensor_value_info("x_post_dynamics", onnx.TensorProto.FLOAT, [3+(i-1)*2])
        step_dynamics_graph = helper.make_graph(
            nodes=step_dynamics_nodes,
            name="step_dynamics",
            inputs =[input],
            outputs=[output]
        )

        step_dynamics_model = helper.make_model(step_dynamics_graph, producer_name='Feiyang Cai')
    
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

        step_graph = onnx.GraphProto()
        step_graph.CopyFrom(step_model.graph)
        for step_node in step_graph.node:
            step_node.name += f"_step_{step-i}"
            for input_i, _ in enumerate(step_node.input):
                if step_node.input[input_i] == step_graph.input[0].name:
                    step_node.input[input_i] = f"x_step_{step-i}"
                else:
                    input_name = step_node.input[input_i]
                    new_name = input_name + f"_step_{step-i}"
                    step_node.input[input_i] += f"_step_{step-i}"
                    for initializer in step_graph.initializer:
                        if initializer.name == input_name:
                            initializer.name = new_name
            for output_i, _ in enumerate(step_node.output):
                if step_node.output[output_i] == step_graph.output[0].name:
                    step_node.output[output_i] = f"x_step_{step-i+1}"
                else:
                    output_name = step_node.output[output_i]
                    new_name = output_name + f"_step_{step-i}"
                    step_node.output[output_i] += f"_step_{step-i}"
                    for initializer in step_graph.initializer:
                        if initializer.name == output_name:
                            initializer.name = new_name
                    
        system_nodes.extend(step_graph.node)
        system_initializers.extend(step_graph.initializer)
    
    input = helper.make_tensor_value_info("x_step_0", onnx.TensorProto.FLOAT, [1, 2+2*step])
    output = helper.make_tensor_value_info(f"x_step_{step}", onnx.TensorProto.FLOAT, [1, 2])
    system_graph = helper.make_graph(
        nodes=system_nodes,
        initializer=system_initializers,
        name="system",
        inputs =[input],
        outputs=[output]
    )

    system_model = helper.make_model(system_graph, producer_name='Feiyang Cai')
    print(system_model.graph.output)
    onnx.save_model(system_model, f"./models/system_model_{step}_{coeff_p}_{coeff_theta}.onnx")

