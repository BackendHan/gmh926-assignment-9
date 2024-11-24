import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial
from mpl_toolkits.mplot3d import Axes3D

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# 定义激活函数及其导数
def activation(z, function):
    if function == 'tanh':
        return np.tanh(z)
    elif function == 'relu':
        return np.maximum(0, z)
    elif function == 'sigmoid':
        return 1 / (1 + np.exp(-z))
    else:
        raise ValueError("Unknown activation function")

def activation_derivative(a, function):
    if function == 'tanh':
        return 1 - np.power(a, 2)
    elif function == 'relu':
        return np.where(a > 0, 1, 0)
    elif function == 'sigmoid':
        return a * (1 - a)
    else:
        raise ValueError("Unknown activation function")

# 定义简单的 MLP 类
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr  # 学习率
        self.activation_fn = activation  # 激活函数

        # 初始化权重和偏置
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))

        # 用于可视化的激活值和梯度
        self.activations = {}
        self.gradients = {}

    def forward(self, X):
        # 输入层到隐藏层
        z1 = X.dot(self.W1) + self.b1
        a1 = activation(z1, self.activation_fn)

        # 隐藏层到输出层
        z2 = a1.dot(self.W2) + self.b2
        a2 = 1 / (1 + np.exp(-z2))  # 输出层使用 sigmoid 激活

        # 存储激活值
        self.activations['X'] = X
        self.activations['z1'] = z1
        self.activations['a1'] = a1
        self.activations['z2'] = z2
        self.activations['a2'] = a2

        return a2

    def backward(self, X, y):
        m = y.shape[0]  # 样本数量

        # 获取前向传播的激活值
        a1 = self.activations['a1']
        a2 = self.activations['a2']

        # 输出层误差
        dz2 = a2 - y  # 损失对 z2 的导数

        # 计算 W2 和 b2 的梯度
        dW2 = (a1.T).dot(dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # 传播到隐藏层
        da1 = dz2.dot(self.W2.T)
        dz1 = da1 * activation_derivative(a1, self.activation_fn)

        # 计算 W1 和 b1 的梯度
        dW1 = (X.T).dot(dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # 存储梯度
        self.gradients['dW1'] = dW1
        self.gradients['dW2'] = dW2
        self.gradients['db1'] = db1
        self.gradients['db2'] = db2

        # 更新权重和偏置
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

def generate_data(n_samples=100):
    np.random.seed(0)
    # 生成输入数据
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int)  # 圆形边界
    y = y.reshape(-1, 1)
    return X, y

# 可视化更新函数
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()
    # 进行训练步骤
    for _ in range(10):
        mlp.forward(X)
        mlp.backward(X, y)
    # 获取隐藏层特征
    hidden_features = mlp.activations['a1']
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title('Hidden Space Features')

    # 绘制输入空间的决策边界
    xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 100),
                         np.linspace(X[:,1].min()-1, X[:,1].max()+1, 100))
    grid = np.c_[xx.ravel(), yy.ravel()]
    probs = mlp.forward(grid)
    probs = probs.reshape(xx.shape)
    ax_input.contourf(xx, yy, probs, levels=[0, 0.5, 1], alpha=0.2, colors=['blue', 'red'])
    ax_input.scatter(X[:,0], X[:,1], c=y.ravel(), cmap='bwr', edgecolors='k')
    ax_input.set_title('Decision Boundary in Input Space')

    # 绘制梯度可视化
    ax_gradient.set_xlim(-1, 3)
    ax_gradient.set_ylim(-2, 2)
    ax_gradient.axis('off')

    # 神经元的位置
    input_neurons = [(-0.5, 1), (-0.5, -1)]
    hidden_neurons = [(0.5, 1), (0.5, 0), (0.5, -1)]
    output_neuron = (1.5, 0)

    # 绘制神经元
    for x, y_pos in input_neurons:
        circle = plt.Circle((x, y_pos), radius=0.1, fill=True, color='lightgray')
        ax_gradient.add_patch(circle)
    for x, y_pos in hidden_neurons:
        circle = plt.Circle((x, y_pos), radius=0.1, fill=True, color='lightgray')
        ax_gradient.add_patch(circle)
    x, y_pos = output_neuron
    circle = plt.Circle((x, y_pos), radius=0.1, fill=True, color='lightgray')
    ax_gradient.add_patch(circle)

    # 从输入层到隐藏层的边
    for i, (x0, y0) in enumerate(input_neurons):
        for j, (x1, y1) in enumerate(hidden_neurons):
            grad = mlp.gradients['dW1'][i, j]
            linewidth = np.abs(grad) * 1000  # 调整比例以便可视化
            color = 'red' if grad < 0 else 'blue'
            ax_gradient.plot([x0, x1], [y0, y1], linewidth=linewidth, color=color, alpha=0.5)

    # 从隐藏层到输出层的边
    for i, (x0, y0) in enumerate(hidden_neurons):
        x1, y1 = output_neuron
        grad = mlp.gradients['dW2'][i, 0]
        linewidth = np.abs(grad) * 1000
        color = 'red' if grad < 0 else 'blue'
        ax_gradient.plot([x0, x1], [y0, y1], linewidth=linewidth, color=color, alpha=0.5)

    ax_gradient.set_title('Gradients Visualization')

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)
    # 设置可视化
    matplotlib.use('Agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)
    # 创建动画
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)
    # 保存动画为 GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
