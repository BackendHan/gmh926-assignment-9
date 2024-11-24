from flask import Flask, render_template, request, jsonify, send_from_directory
import os
from neural_networks import visualize

app = Flask(__name__)

# Define the main route
@app.route('/')
def index():
    return render_template('index.html')

# Route to handle experiment parameters and trigger the experiment
# 修改 /run_experiment 路由
@app.route('/run_experiment', methods=['POST'])
def run_experiment():
    try:
        activation = request.json['activation']
        lr = float(request.json['lr'])
        step_num = int(request.json['step_num'])

        # 参数校验
        if activation not in ['tanh', 'relu', 'sigmoid']:
            return jsonify({"error": "Invalid activation function. Choose from tanh, relu, sigmoid."}), 400
        if lr <= 0:
            return jsonify({"error": "Learning rate must be positive."}), 400
        if step_num <= 0:
            return jsonify({"error": "Number of training steps must be a positive integer."}), 400

        # 在主线程中完成动画渲染
        visualize(activation, lr, step_num)

        # 返回结果GIF路径
        result_gif = "results/visualize.gif"
        return jsonify({"result_gif": result_gif if os.path.exists(result_gif) else None})
    except Exception as e:
        return jsonify({"error": str(e)}), 500




# Route to serve result images
@app.route('/results/<filename>')
def results(filename):
    return send_from_directory('results', filename)

if __name__ == '__main__':
    app.run(debug=True)