import random
import math

INPUT_SIZE = 1  
HIDDEN_SIZE = 12  
OUTPUT_SIZE = 1  
LEARNING_RATE = 0.01  
EPOCHS = 5000  
LEAKY_RELU_ALPHA = 0.01 
PRINT_EVERY = 100
DATA_FILE = "data.txt"

def init_weights(rows, cols):
    return [[random.uniform(-0.5, 0.5) for _ in range(cols)] for _ in range(rows)]

weights1 = init_weights(INPUT_SIZE, HIDDEN_SIZE)
bias1 = [random.uniform(-0.5, 0.5) for _ in range(HIDDEN_SIZE)]

weights2 = init_weights(HIDDEN_SIZE, HIDDEN_SIZE)
bias2 = [random.uniform(-0.5, 0.5) for _ in range(HIDDEN_SIZE)]

weights3 = init_weights(HIDDEN_SIZE, OUTPUT_SIZE)
bias3 = [random.uniform(-0.5, 0.5) for _ in range(OUTPUT_SIZE)]

def leaky_relu(x):
    return x if x > 0 else LEAKY_RELU_ALPHA * x

def leaky_relu_derivative(x):
    return 1 if x > 0 else LEAKY_RELU_ALPHA

def forward(x):
    hidden1 = [leaky_relu(sum(x[i] * weights1[i][j] for i in range(INPUT_SIZE)) + bias1[j]) for j in range(HIDDEN_SIZE)]
    hidden2 = [leaky_relu(sum(hidden1[i] * weights2[i][j] for i in range(HIDDEN_SIZE)) + bias2[j]) for j in range(HIDDEN_SIZE)]
    output = [sum(hidden2[i] * weights3[i][j] for i in range(HIDDEN_SIZE)) + bias3[j] for j in range(OUTPUT_SIZE)]
    return hidden1, hidden2, output

def backward(x, y, hidden1, hidden2, output):
    global weights1, weights2, weights3, bias1, bias2, bias3
    output_error = [output[i] - y[i] for i in range(OUTPUT_SIZE)]
    output_gradient = output_error
    
    hidden2_error = [sum(output_gradient[j] * weights3[i][j] for j in range(OUTPUT_SIZE)) for i in range(HIDDEN_SIZE)]
    hidden2_gradient = [hidden2_error[i] * leaky_relu_derivative(hidden2[i]) for i in range(HIDDEN_SIZE)]
    
    hidden1_error = [sum(hidden2_gradient[j] * weights2[i][j] for j in range(HIDDEN_SIZE)) for i in range(HIDDEN_SIZE)]
    hidden1_gradient = [hidden1_error[i] * leaky_relu_derivative(hidden1[i]) for i in range(HIDDEN_SIZE)]
    
    for i in range(HIDDEN_SIZE):
        for j in range(OUTPUT_SIZE):
            weights3[i][j] -= LEARNING_RATE * output_gradient[j] * hidden2[i]
    for j in range(OUTPUT_SIZE):
        bias3[j] -= LEARNING_RATE * output_gradient[j]
    
    for i in range(HIDDEN_SIZE):
        for j in range(HIDDEN_SIZE):
            weights2[i][j] -= LEARNING_RATE * hidden2_gradient[j] * hidden1[i]
    for j in range(HIDDEN_SIZE):
        bias2[j] -= LEARNING_RATE * hidden2_gradient[j]
    
    for i in range(INPUT_SIZE):
        for j in range(HIDDEN_SIZE):
            weights1[i][j] -= LEARNING_RATE * hidden1_gradient[j] * x[i]
    for j in range(HIDDEN_SIZE):
        bias1[j] -= LEARNING_RATE * hidden1_gradient[j]

def load_training_data():
    data = []
    with open(DATA_FILE, "r") as f:
        for line in f:
            x, y = map(float, line.split())
            data.append(((x,), (y,)))
    return data

training_data = load_training_data()

for epoch in range(EPOCHS):
    total_loss = 0
    for x, y in training_data:
        hidden1, hidden2, output = forward(x)
        loss = sum((output[i] - y[i])**2 for i in range(OUTPUT_SIZE)) / OUTPUT_SIZE
        total_loss += loss
        backward(x, y, hidden1, hidden2, output)
    
    if (epoch + 1) % PRINT_EVERY == 0:
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {total_loss / len(training_data):.6f}")

test_data = []
with open("data.txt", "r") as file:
    for line in file:
        x, y = map(float, line.strip().split())
        test_data.append((x, y))

print("\ntest:")
for x, true_y in test_data:
    _, _, y_pred = forward((x,))
    print(f"x = {x:.10f}, pred = {y_pred[0]:.10f}, skutecna hodnota y = {true_y:.10f}")

while True:
    user_input = input("enter your x = ")
    if user_input.lower() == 'q':
        break
    try:
        x = float(user_input)
        _, _, y_pred = forward((x,))
        print(f"pred x = {x:.10f}: {y_pred[0]:.10f}")
    except ValueError:
        print("err input")
