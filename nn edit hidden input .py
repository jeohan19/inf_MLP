import random

INPUT_SIZE = 1  
NUM_HIDDEN_LAYERS = 4  
HIDDEN_SIZE = 32  
OUTPUT_SIZE = 1  
LEARNING_RATE = 0.001  
EPOCHS = 1000  
LEAKY_RELU_ALPHA = 0.01  
PRINT_EVERY = 100
DATA_FILE = "data.txt"

def init_weights(rows, cols):
    return [[random.uniform(-0.5, 0.5) for _ in range(cols)] for _ in range(rows)]

weights = [init_weights(INPUT_SIZE, HIDDEN_SIZE)]
biases = [[random.uniform(-0.5, 0.5) for _ in range(HIDDEN_SIZE)]]

for _ in range(NUM_HIDDEN_LAYERS - 1):
    weights.append(init_weights(HIDDEN_SIZE, HIDDEN_SIZE))
    biases.append([random.uniform(-0.5, 0.5) for _ in range(HIDDEN_SIZE)])

weights.append(init_weights(HIDDEN_SIZE, OUTPUT_SIZE))
biases.append([random.uniform(-0.5, 0.5) for _ in range(OUTPUT_SIZE)])

def leaky_relu(x):
    return x if x > 0 else LEAKY_RELU_ALPHA * x

def leaky_relu_derivative(x):
    return 1 if x > 0 else LEAKY_RELU_ALPHA

def forward(x):
    layers = [x]
    for i in range(NUM_HIDDEN_LAYERS + 1):
        next_layer = [leaky_relu(sum(layers[-1][k] * weights[i][k][j] for k in range(len(layers[-1]))) + biases[i][j])
                      for j in range(len(biases[i]))]
        layers.append(next_layer)
    return layers

def backward(x, y, layers):
    global weights, biases
    errors = [ [layers[-1][i] - y[i] for i in range(OUTPUT_SIZE)] ]
    gradients = [errors[0]]
    
    for i in range(NUM_HIDDEN_LAYERS, -1, -1):
        layer_errors = [sum(gradients[0][j] * weights[i][k][j] for j in range(len(gradients[0]))) for k in range(len(layers[i]))]
        gradients.insert(0, [layer_errors[k] * leaky_relu_derivative(layers[i][k]) for k in range(len(layer_errors))])
        
        for j in range(len(gradients[1])):
            for k in range(len(layers[i])):
                weights[i][k][j] -= LEARNING_RATE * gradients[1][j] * layers[i][k]
            biases[i][j] -= LEARNING_RATE * gradients[1][j]

def load_training_data():
    data = []
    with open(DATA_FILE, "r") as f:
        for line in f:
            values = list(map(float, line.split()))
            x = values[:-1]  
            y = values[-1:]  
            data.append((x, y))
    return data

training_data = load_training_data()

for epoch in range(EPOCHS):
    total_loss = 0
    for x, y in training_data:
        layers = forward(x)
        loss = sum((layers[-1][i] - y[i])**2 for i in range(OUTPUT_SIZE)) / OUTPUT_SIZE
        total_loss += loss
        backward(x, y, layers)
    
    if (epoch + 1) % PRINT_EVERY == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(training_data):.6f}")

print("\ntest:")
for x, true_y in training_data:
    y_pred = forward(x)[-1]
    print(f"x = {x}, pred = {y_pred[0]:.10f}, skutecna hodnota y = {true_y[0]:.10f}")

while True:
    user_input = input("enter your x = ")
    if user_input.lower() == 'q':
        break
    try:
        x = list(map(float, user_input.split()))
        if len(x) != INPUT_SIZE:
            print(f"err: expected num of inputs: {INPUT_SIZE}")
            continue
        y_pred = forward(x)[-1]
        print(f"pred: {y_pred[0]:.10f}")
    except ValueError:
        print("err input")
