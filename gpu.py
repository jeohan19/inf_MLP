import torch_directml
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import torch
import time

# ✅ Přesun na GPU přes DirectML, pokud je k dispozici
device = torch_directml.device()
print(f"Použité zařízení: {device}")

# ✅ Parametry sítě
INPUT_SIZE = 1
NUM_HIDDEN_LAYERS = 32
HIDDEN_SIZE = 128
OUTPUT_SIZE = 1
LEARNING_RATE = 0.0001
EPOCHS = 10000
LEAKY_RELU_ALPHA = 0.01
PRINT_EVERY = 1
DATA_FILE = "train/training_data_1.txt"
TEST_DATA = "train/training_data_1.txt"

# ✅ Definice neuronové sítě v PyTorch
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_hidden_layers, alpha):
        super(MLP, self).__init__()
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.LeakyReLU(negative_slope=alpha))
        for _ in range(num_hidden_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.LeakyReLU(negative_slope=alpha))
        layers.append(nn.Linear(hidden_size, output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# ✅ Načtení trénovacích dat
def load_data(file):
    data = np.loadtxt(file)
    x = torch.tensor(data[:, :-1], dtype=torch.float32).to(device)
    y = torch.tensor(data[:, -1:], dtype=torch.float32).to(device)
    return x, y

# ✅ Inicializace modelu, ztrátové funkce a optimalizátoru
model = MLP(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, NUM_HIDDEN_LAYERS, LEAKY_RELU_ALPHA).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# ✅ Načtení dat
train_x, train_y = load_data(DATA_FILE)
test_x, test_y = load_data(TEST_DATA)

# ✅ Trénování modelu
time_start = time.time()
loss_history = []
for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad()
    outputs = model(train_x)
    loss = criterion(outputs, train_y)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % PRINT_EVERY == 0:
        loss_history.append(loss.item())
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {loss.item():.6f}")
time_end = time.time()  

# ✅ Testování modelu
model.eval()
with torch.no_grad():
    predictions = model(test_x).cpu().numpy()
    true_values = test_y.cpu().numpy()

# ✅ Vizualizace ztráty
plt.figure(figsize=(10, 5))
plt.plot(loss_history, label="Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Křivka učení (DirectML akcelerace)")
plt.legend()
plt.show()

# ✅ Porovnání predikcí
plt.figure(figsize=(10, 5))
plt.scatter(test_x.cpu().numpy(), true_values, color='blue', label='Skutečné hodnoty')
plt.scatter(test_x.cpu().numpy(), predictions, color='red', label='Predikce')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title("Porovnání skutečných a predikovaných hodnot")
plt.show()

print(time_end - time_start)    