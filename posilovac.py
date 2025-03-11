import numpy as np
import gym
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# Aktivace DirectML
import torch_directml
device = torch_directml.device()

# Parametry prostředí a modelu
env = gym.make("FrozenLake-v1", is_slippery=False)
state_size = env.observation_space.n  # Počet stavů (16 pro FrozenLake)
action_size = env.action_space.n  # Počet možných akcí
gamma = 0.95  # Slevovací faktor
learning_rate = 0.001

# Vytvoření modelu
class DQN(nn.Module):
    def __init__(self, input_size, output_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_size, 24)
        self.fc2 = nn.Linear(24, 24)
        self.fc3 = nn.Linear(24, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Funkce pro epsilon-greedy politiku
def epsilon_greedy_policy(state, epsilon, model):
    if np.random.rand() <= epsilon:
        return np.random.choice(action_size)  # Náhodná akce
    else:
        with torch.no_grad():
            q_values = model(state)
        return torch.argmax(q_values).item()  # Nejlepší akce

# Trénovací funkce
def train_dqn(episodes=1000):
    model = DQN(state_size, action_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    epsilon = 1.0  # Začínáme s náhodnými akcemi
    epsilon_decay = 0.995  # Snižování epsilon v průběhu času
    epsilon_min = 0.01  # Minimální hodnota epsilon
    total_rewards = []

    for episode in range(episodes):
        state = env.reset()  # Resetování prostředí
        state = state[0] if isinstance(state, tuple) else state  # Pokud je state v tuple, vezmeme první prvek

        # One-hot kódování stavu
        state = np.eye(state_size)[state]  # Převod na one-hot vektor
        state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)  # Přeformátování stavu pro model

        done = False
        total_reward = 0

        while not done:
            action = epsilon_greedy_policy(state, epsilon, model)
            next_state, reward, done, _, _ = env.step(action)

            # One-hot kódování pro next_state
            next_state = np.eye(state_size)[next_state]  # Převod na one-hot vektor
            next_state = torch.tensor(next_state, dtype=torch.float32).unsqueeze(0).to(device)  # Přeformátování na [1, state_size]
            
            # Učení: Q(s, a) = r + gamma * max(Q(s', a'))
            with torch.no_grad():
                target = reward + gamma * torch.max(model(next_state)) if not done else reward
            target_f = model(state)
            target_f[0][action] = target

            # Trénování modelu
            optimizer.zero_grad()
            loss = nn.MSELoss()(target_f, target_f.detach())
            loss.backward()
            optimizer.step()
            
            state = next_state
            total_reward += reward

        epsilon = max(epsilon_min, epsilon * epsilon_decay)  # Aktualizace epsilon pro další epizodu
        total_rewards.append(total_reward)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{episodes} - Reward: {total_reward}, Epsilon: {epsilon}")

    # Výsledky
    plt.plot(total_rewards)
    plt.title("Total Rewards over Episodes")
    plt.show()

    return model

# Spuštění trénování
model = train_dqn()
