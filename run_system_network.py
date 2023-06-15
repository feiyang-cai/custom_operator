import numpy as np
import onnxruntime as rt

p_range = [-6, -5]
theta_range = [0, 30]
p_num_bin = 8
theta_num_bin = 74
p_bins = np.linspace(p_range[0], p_range[1], p_num_bin+1, endpoint=True)
p_lbs = np.array(p_bins[:-1],dtype=np.float32)
p_ubs = np.array(p_bins[1:], dtype=np.float32)

theta_bins = np.linspace(theta_range[0], theta_range[1], theta_num_bin+1, endpoint=True)
theta_lbs = np.array(theta_bins[:-1],dtype=np.float32)
theta_ubs = np.array(theta_bins[1:], dtype=np.float32)

p_choices = 2
theta_choices = 1

p_lb = p_lbs[p_choices]
p_ub = p_ubs[p_choices]

theta_lb = theta_lbs[theta_choices]
theta_ub = theta_ubs[theta_choices]
    
samples = 1000
z = np.random.uniform(-0.8, 0.8, size=(samples, 2)).astype(np.float32)
p = np.random.uniform(p_lb, p_ub, size=(samples, 1)).astype(np.float32)
theta = np.random.uniform(theta_lb, theta_ub, size=(samples, 1)).astype(np.float32)

shared_library = "./lib/libcustom_dynamics.so"
so = rt.SessionOptions()
so.register_custom_ops_library(shared_library)

model_path = './models/system_model.onnx'
sess = rt.InferenceSession(model_path, so)

input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
output_name = sess.get_outputs()[0].name

y = []
for z_i, p_i, theta_i in zip(z, p, theta):
    input_0 = np.concatenate([z_i, p_i, theta_i]).astype(np.float32).reshape(input_shape)
    res = sess.run([output_name], {input_name: input_0})
    y.append(res[0][0])
y = np.array(y)

from matplotlib import pyplot as plt
fig, ax = plt.subplots()
ax.scatter(y[:, 2], y[:, 3], s=1, alpha=0.5)
## plot grids
for p_lb in p_lbs:
    X = [p_lb, p_lb]
    Y = [theta_bins[0], theta_bins[-1]]
    ax.plot(X, Y, 'lightgray', alpha=0.2)

for theta_lb in theta_lbs:
    Y = [theta_lb, theta_lb]
    X = [p_bins[0], p_bins[-1]]
    ax.plot(X, Y, 'lightgray', alpha=0.2)

ax.set_xticks(p_bins)
ax.set_yticks(theta_bins)
ax.set_xticks([-6, -5.8, -5.6, -5.4, -5.2, -5])
ax.set_yticks([0, 2, 4, 6])
ax.set_xlim([-6, -5])
ax.set_ylim([0, 7])
ax.set_xlabel(r"$p$ (m)")
ax.set_ylabel(r"$\theta$ (degrees)")

plt.savefig('system_network.png', dpi=300)