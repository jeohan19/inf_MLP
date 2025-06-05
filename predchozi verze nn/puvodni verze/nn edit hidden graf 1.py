import random
import math
##########
#analyse
import matplotlib.pyplot as plt
import numpy as np
import time
import networkx as nx
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import torch
##########

INPUT_SIZE = 1
NUM_HIDDEN_LAYERS = 4
HIDDEN_SIZE = 16
OUTPUT_SIZE = 1
LEARNING_RATE = 0.001
EPOCHS = 150
LEAKY_RELU_ALPHA = 0.1
PRINT_EVERY = 1
FUNKCE = "cosine: y = cos(x)"
ROZSAH_TRAIN_DAT = "-8 8"
DATA_FILE = "train\\tr_cosine16.txt"
TEST_DATA = "train\\te_cosine16.txt"





##################################################################################################
def draw_neural_network(input_size, hidden_layers, hidden_size, output_size):
    G = nx.DiGraph()
    positions = {}
    labels = {}
    layer_sizes = [input_size] + [hidden_size] * hidden_layers + [output_size]
    
    neuron_id = 0
    x_spacing = 2  
    y_spacing = 1.5
    
    for layer_idx, layer_size in enumerate(layer_sizes):
        y_offset = - (layer_size - 1) * y_spacing / 2
        
        for neuron_idx in range(layer_size):
            G.add_node(neuron_id)
            positions[neuron_id] = (layer_idx * x_spacing, y_offset + neuron_idx * y_spacing)
            labels[neuron_id] = f'n{neuron_id+1}'
            neuron_id += 1
    
    prev_layer_start = 0
    for layer_idx in range(len(layer_sizes) - 1):
        current_layer_start = prev_layer_start + layer_sizes[layer_idx]
        
        for i in range(layer_sizes[layer_idx]):
            for j in range(layer_sizes[layer_idx + 1]):
                G.add_edge(prev_layer_start + i, current_layer_start + j)
        
        prev_layer_start = current_layer_start
    
    plt.figure(figsize=(19, 10))
    nx.draw(G, pos=positions, with_labels=True, labels=labels, node_color='lightblue', edge_color='gray', node_size=1500)
    plt.suptitle(f"Struktura neuronové sítě\nNeurony ve skrytých vrstvách: {hidden_size}, Skryté vrstvy: {hidden_layers}")
    plt.subplots_adjust(top=3)
    plt.show()
############################################################################################


def he_init_weights(rows, cols):
    """Inicializace vah pomocí He normal (vhodné pro ReLU a Leaky ReLU)"""
    std_dev = math.sqrt(2 / rows)
    return [[random.gauss(0, std_dev) for _ in range(cols)] for _ in range(rows)]

def init_biases(size):
    """Inicializace biasů na nulu (doporučené pro He inicializaci)"""
    return [0.0 for _ in range(size)]

# Inicializace vah a biasů pro vstupní vrstvu do skryté vrstvy
weights = [he_init_weights(INPUT_SIZE, HIDDEN_SIZE)]
biases = [init_biases(HIDDEN_SIZE)]

# Inicializace pro skryté vrstvy
for _ in range(NUM_HIDDEN_LAYERS - 1):
    weights.append(he_init_weights(HIDDEN_SIZE, HIDDEN_SIZE))
    biases.append(init_biases(HIDDEN_SIZE))

# Inicializace pro poslední skrytou vrstvu do výstupní vrstvy
weights.append(he_init_weights(HIDDEN_SIZE, OUTPUT_SIZE))
biases.append(init_biases(OUTPUT_SIZE))



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
    global errors 
    errors = [ [layers[-1][i] - y[i] for i in range(OUTPUT_SIZE)] ]
    gradients = [errors[0]]
    
    for i in range(NUM_HIDDEN_LAYERS, -1, -1):
        layer_errors = [sum(gradients[0][j] * weights[i][k][j] for j in range(len(gradients[0]))) for k in range(len(layers[i]))]
        gradients.insert(0, [layer_errors[k] * leaky_relu_derivative(layers[i][k]) for k in range(len(layer_errors))])
        
        for j in range(len(gradients[1])):
            for k in range(len(layers[i])):
                weights[i][k][j] -= LEARNING_RATE * gradients[1][j] * layers[i][k]
            biases[i][j] -= LEARNING_RATE * gradients[1][j]

def load_training_data(file):
    data = []
    with open(file, "r") as f:
        for line in f:
            values = list(map(float, line.split()))
            x = values[:-1]  
            y = values[-1:]  
            data.append((x, y))
    return data

training_data = load_training_data(DATA_FILE)
gradients_history = []
loss_history = []

####################### pokus procentualnmi odchylka
def absolute_percentage_error(y_true, y_pred):
    return abs((y_true - y_pred) / y_true) * 100

def calculate_mean_absolute_percentage_error(true_values, predictions):
    errors = [absolute_percentage_error(true, pred) for true, pred in zip(true_values, predictions)]
    return sum(errors) / len(errors)

def mean_squared_error(y_true, y_pred):
    """Vypočítá průměrnou kvadratickou chybu (MSE)"""
    return sum((y_true[i] - y_pred[i]) ** 2 for i in range(len(y_true))) / len(y_true)

#######################

#struct





draw_neural_network(INPUT_SIZE, NUM_HIDDEN_LAYERS, HIDDEN_SIZE, OUTPUT_SIZE)

start_time = time.time()

for epoch in range(EPOCHS):
    random.shuffle(training_data)
    total_loss = 0
    for x, y in training_data:
        layers = forward(x)
        #loss = sum((round(layers[-1][i] - y[i], 16) ** 2) for i in range(OUTPUT_SIZE)) / OUTPUT_SIZE
        loss = mean_squared_error(y, layers[-1])

        total_loss += loss
        backward(x, y, layers)
        gradients_history.append(errors)
    
    if (epoch + 1) % PRINT_EVERY == 0:
        avg_loss = total_loss / len(training_data)
        loss_history.append(avg_loss)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {avg_loss:.6f}")

end_time = time.time()



'''
for i in weights:
    print(i)
for j in biases:
    print(j)
'''


'''''
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

 '''''

print("\ntest:")
predictions = []
true_values = []
x_values = []
test_data = load_training_data(TEST_DATA)
random.shuffle(test_data)

for x, true_y in test_data:
    y_pred = forward(x)[-1]
    predictions.append(y_pred[0])
    true_values.append(true_y[0])
    x_values.append(x[0])

    #####################
    # Výpočet průměrné procentuální odchylky
mean_absolute_percentage_error = calculate_mean_absolute_percentage_error(true_values, predictions)
print(f"Mean Absolute Percentage Error (MAPE): {mean_absolute_percentage_error:.2f}%")
elapsed_time = (end_time - start_time)/60
print(f"Elapsed time: {elapsed_time:.2f}minutes")
    ################x####

x_plot = np.array(x_values)
y_true_plot = np.array(true_values)
y_pred_plot = np.array(predictions)

plt.figure(figsize=(19, 10))
plt.plot(range(PRINT_EVERY, EPOCHS + 1, PRINT_EVERY), loss_history, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Křivka učení\nneurony ve skrytych vrstvách: {HIDDEN_SIZE}, skryté vrstvy: {NUM_HIDDEN_LAYERS}, alpha: {LEAKY_RELU_ALPHA}, learning_rate: {LEARNING_RATE} \nfunkce: {FUNKCE}, rozsah train dat: {ROZSAH_TRAIN_DAT}")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(19, 10))
plt.scatter(x_plot, y_true_plot, color='blue', label='skutečné hodnoty')
plt.scatter(x_plot, y_pred_plot, color='red', label='predikované hodnoty')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title(f"porovnání skutečné a naučené funkce\nneurony ve skrytych vrstvách: {HIDDEN_SIZE}, skryté vrstvy: {NUM_HIDDEN_LAYERS}, alpha: {LEAKY_RELU_ALPHA}, learning_rate: {LEARNING_RATE} \n funkce: {FUNKCE}, rozsah train dat: {ROZSAH_TRAIN_DAT}")
plt.grid(True)
plt.show()

print("complete")

plt.figure(figsize=(19, 10))
sns.histplot(np.array(true_values) - np.array(predictions), bins=30, kde=True)
plt.xlabel("Chyba (y_true - y_pred)")
plt.ylabel("Počet vzorků")
plt.title("Histogram chyb")
plt.grid(True)
plt.show()

plt.figure(figsize=(19, 10))
sns.kdeplot(true_values, label="Skutečné hodnoty", fill=True, color="blue")
sns.kdeplot(predictions, label="Predikované hodnoty", fill=True, color="red")
plt.xlabel("y")
plt.ylabel("Hustota pravděpodobnosti")
plt.legend()
plt.title("Porovnání distribuce skutečných a predikovaných hodnot")
plt.grid(True)
plt.show()
