import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt

# Load telemetry data
telemetry_data = pd.read_csv("data/satellite_telemetry_advanced.csv")

# RL Environment
class SatelliteEnv:
    def __init__(self, telemetry_data):
        self.data = telemetry_data
        self.num_steps = len(telemetry_data)
        self.current_step = 0

    def reset(self):
        self.current_step = 0
        return self._get_state()

    def _get_state(self):
        row = self.data.iloc[self.current_step]
        return np.array([row["x"], row["y"], row["z"], row["vx"], row["vy"], row["vz"], row["battery"]])

    def step(self, action):
        # Simulate the effect of action (e.g., adjusting velocity)
        done = False
        self.current_step += 1

        # Reward logic (e.g., maintain orbit stability and battery level)
        state = self._get_state()
        reward = -np.linalg.norm(state[:3]) / 1e6  # Penalize large deviations
        reward += state[-1] / 100  # Reward for maintaining battery life

        if self.current_step >= self.num_steps - 1 or state[-1] <= 0:
            done = True  # End episode if out of data or battery depleted

        return state, reward, done

# Neural Network for DQN
class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# DQN Training with Metric Tracking
def train_dqn(env, num_episodes=500, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.995, lr=0.001):
    state_dim = 7  # State dimensions (x, y, z, vx, vy, vz, battery)
    action_dim = 3  # Actions (e.g., thrust up, thrust down, no action)

    q_network = DQNetwork(state_dim, action_dim)
    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    replay_buffer = deque(maxlen=10000)
    batch_size = 64

    # Metrics to track
    episode_rewards = []
    training_losses = []
    epsilons = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)
            else:
                with torch.no_grad():
                    q_values = q_network(torch.tensor(state, dtype=torch.float32))
                    action = torch.argmax(q_values).item()

            # Take action in the environment
            next_state, reward, done = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))
            total_reward += reward
            state = next_state

            # Sample and train the network
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards_, next_states, dones = zip(*batch)

                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.long)
                rewards_ = torch.tensor(rewards_, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                # Q-value prediction and target calculation
                q_values = q_network(states)
                q_values = q_values.gather(1, actions.unsqueeze(-1)).squeeze(-1)

                next_q_values = q_network(next_states).max(1)[0]
                targets = rewards_ + gamma * next_q_values * (1 - dones)

                # Compute loss and backpropagation
                loss = loss_fn(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Track training loss
                training_losses.append(loss.item())

        # Track episode metrics
        episode_rewards.append(total_reward)
        epsilons.append(epsilon)

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        print(f"Episode {episode + 1}/{num_episodes}, Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

    return q_network, episode_rewards, training_losses, epsilons

# Train the model
env = SatelliteEnv(telemetry_data)
q_network, episode_rewards, training_losses, epsilons = train_dqn(env)


# Save the model
torch.save(q_network.state_dict(), "dqn_satellite_model.pth")

# Plotting metrics
# 1. Total rewards per episode
plt.figure(figsize=(10, 6))
plt.plot(episode_rewards, label="Total Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Episode Rewards During Training")
plt.legend()
plt.grid()
plt.savefig("figures/reward_per_episode.png", dpi=300)
plt.close()

# 2. Training loss over steps
plt.figure(figsize=(10, 6))
plt.plot(training_losses, label="Training Loss")
plt.xlabel("Training Steps")
plt.ylabel("Loss")
plt.title("Training Loss Over Time")
plt.legend()
plt.grid()
plt.savefig("figures/training_loss.png", dpi=300)
plt.close()

# 3. Epsilon decay over episodes
plt.figure(figsize=(10, 6))
plt.plot(epsilons, label="Epsilon Decay")
plt.xlabel("Episode")
plt.ylabel("Epsilon")
plt.title("Epsilon Decay During Training")
plt.legend()
plt.grid()
plt.savefig("figures/epsilon_decay.png", dpi=300)
plt.close()