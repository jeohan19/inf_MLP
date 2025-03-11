import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection

# Parametry neuronové sítě
INPUT_SIZE = 2
HIDDEN_SIZE = 4
NUM_HIDDEN_LAYERS = 2
OUTPUT_SIZE = 1
EPOCHS = 50
SPEED = 100

# Testovací data
test_data = [((x,), (np.sin(x),)) for x in np.linspace(-8, 8, 50)]
x_values = np.array([x[0] for x, _ in test_data])
true_values = np.array([true_y[0] for _, true_y in test_data])
epoch_predictions = [true_values + np.random.randn(*true_values.shape) * 0.1 for _ in range(EPOCHS)]

# Simulace historie loss hodnot
loss_history = np.exp(-np.linspace(0, 4, EPOCHS)) * 0.5
PRINT_EVERY = 1

# Simulace vah pro vizualizaci neuronové sítě
epoch_weights = [np.random.uniform(-1, 1, size=(NUM_HIDDEN_LAYERS + 1, HIDDEN_SIZE, HIDDEN_SIZE)) for _ in range(EPOCHS)]

# Vytvoření figure s maticí 2×2
fig, axs = plt.subplots(2, 2, figsize=(20, 12))
fig.suptitle("Vizualizace trénování neuronové sítě")

### 1. Graf: Struktura neuronové sítě ###
def draw_neural_network(ax):
    G = nx.DiGraph()
    positions = {}
    labels = {}
    layer_sizes = [INPUT_SIZE] + [HIDDEN_SIZE] * NUM_HIDDEN_LAYERS + [OUTPUT_SIZE]
    
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

    ax.set_title("Struktura neuronové sítě")
    nx.draw(G, pos=positions, with_labels=True, labels=labels, node_color='lightblue', edge_color='gray', node_size=1500, ax=ax)

draw_neural_network(axs[0, 0])

### 2. Graf: Predikce během trénování ###
ax_pred = axs[0, 1]
line_true, = ax_pred.plot(x_values, true_values, 'bo', label='Skutečné hodnoty')
line_pred, = ax_pred.plot([], [], 'ro', label='Predikované hodnoty')
ax_pred.set_xlim(min(x_values), max(x_values))
ax_pred.set_ylim(min(true_values) - 1, max(true_values) + 1)
ax_pred.set_xlabel('x')
ax_pred.set_ylabel('y')
ax_pred.set_title("Predikce během trénování")
ax_pred.legend()

def update_pred(epoch):
    predictions_epoch = epoch_predictions[epoch]
    line_pred.set_data(x_values, predictions_epoch)
    return line_pred,

ani_pred = animation.FuncAnimation(fig, update_pred, frames=EPOCHS, interval=SPEED, blit=False)

### 3. Graf: Vývoj vah neuronové sítě ###
ax_weights = axs[1, 0]
norm = Normalize(vmin=-1, vmax=1)
sm = ScalarMappable(cmap=plt.cm.seismic, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax_weights)
cbar.set_label('Hodnota váhy')

# Připravení hran a uzlů
positions = {}
neuron_id = 0
x_spacing, y_spacing = 2, 1.5
layer_sizes = [INPUT_SIZE] + [HIDDEN_SIZE] * NUM_HIDDEN_LAYERS + [OUTPUT_SIZE]

for layer_idx, layer_size in enumerate(layer_sizes):
    y_offset = -(layer_size - 1) * y_spacing / 2
    for neuron_idx in range(layer_size):
        positions[neuron_id] = (layer_idx * x_spacing, y_offset + neuron_idx * y_spacing)
        neuron_id += 1

edges = []
edge_segments = []
prev_layer_start = 0
for layer_idx in range(len(layer_sizes) - 1):
    current_layer_start = prev_layer_start + layer_sizes[layer_idx]
    for i in range(layer_sizes[layer_idx]):
        for j in range(layer_sizes[layer_idx + 1]):
            edges.append((prev_layer_start + i, current_layer_start + j))
            edge_segments.append([positions[prev_layer_start + i], positions[current_layer_start + j]])
    prev_layer_start = current_layer_start

edge_collection = LineCollection(edge_segments, cmap=plt.cm.seismic, norm=norm, linewidths=2)
ax_weights.add_collection(edge_collection)
ax_weights.set_title("Vývoj vah neuronové sítě")

def update_weights(epoch):
    weights_list = np.random.uniform(-1, 1, size=len(edges))
    edge_collection.set_array(weights_list)
    edge_collection.set_linewidths(np.abs(weights_list) * 2)
    return edge_collection,

ani_weights = animation.FuncAnimation(fig, update_weights, frames=EPOCHS, interval=SPEED, blit=False)

### 4. Graf: Ztrátová funkce ###
ax_loss = axs[1, 1]
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss")
ax_loss.set_title("Vývoj ztrátové funkce")
ax_loss.grid(True)
loss_line, = ax_loss.plot([], [], label="Loss", color="blue")
ax_loss.set_xlim(0, EPOCHS)
ax_loss.set_ylim(0, max(loss_history) * 1.1)
ax_loss.legend()

def update_loss(epoch_idx):
    x_data = np.arange(PRINT_EVERY, (epoch_idx + 1) * PRINT_EVERY + 1, PRINT_EVERY)
    y_data = loss_history[:len(x_data)]
    loss_line.set_data(x_data, y_data)
    return loss_line,

ani_loss = animation.FuncAnimation(fig, update_loss, frames=len(loss_history), interval=SPEED, blit=False)

plt.show()
