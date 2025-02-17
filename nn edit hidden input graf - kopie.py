import random
import math
import matplotlib.pyplot as plt
import numpy as np
import time
import seaborn as sns
from sklearn.manifold import TSNE
import torch
import torch.nn as nn
import torch.optim as optim

# Parametry modelu
INPUT_SIZE = 1
NUM_HIDDEN_LAYERS = 4
HIDDEN_SIZE = 32
OUTPUT_SIZE = 1
LEARNING_RATE = 0.0001
EPOCHS = 150
LEAKY_RELU_ALPHA = 0.01

# Data a cesty k souborům
TRAIN_DATA_FILE = "train\\tr_x_cos_x216.txt"
TEST_DATA_FILE = "train\\te_x_cos_x216.txt"

# Inicializace modelu
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(INPUT_SIZE, HIDDEN_SIZE),
            nn.LeakyReLU(LEAKY_RELU_ALPHA),
            *[nn.Sequential(nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE), nn.LeakyReLU(LEAKY_RELU_ALPHA)) for _ in range(NUM_HIDDEN_LAYERS - 1)],
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE),
            nn.Linear(HIDDEN_SIZE, OUTPUT_SIZE)
        )

    def forward(self, x):
        return self.layers(x)

# Načtení dat
def load_data(file_path):
    data = np.loadtxt(file_path)
    return torch.tensor(data[:, :-1], dtype=torch.float32), torch.tensor(data[:, -1:], dtype=torch.float32)

# Trénovací a testovací data
train_x, train_y = load_data(TRAIN_DATA_FILE)
test_x, test_y = load_data(TEST_DATA_FILE)

# Model, loss a optimizer
model = NeuralNetwork().to('cuda')
loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Trénování modelu
def train_model(model, train_x, train_y, epochs):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(train_x.cuda())
        loss = loss_function(outputs, train_y.cuda())
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

# Testování modelu
def test_model(model, test_x, test_y):
    model.eval()
    with torch.no_grad():
        predictions = model(test_x.cuda())
        mse = loss_function(predictions, test_y.cuda())
        print(f"Test MSE: {mse.item():.4f}")

# Hlavní běh
start_time = time.time()
train_model(model, train_x, train_y, EPOCHS)
test_model(model, test_x, test_y)
end_time = time.time()
