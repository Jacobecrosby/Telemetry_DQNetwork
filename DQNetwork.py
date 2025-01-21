import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
import matplotlib.pyplot as plt
import logging
import os
from datetime import datetime
from module import setup_logger



now = datetime.now()
date = now.strftime("%Y-%m-%d")

log_directory = f'logs/{date}'
os.makedirs(log_directory, exist_ok=True)

# Load telemetry data
telemetry_data = pd.read_csv("data/satellite_telemetry_leo.csv")

logger = setup_logger(log_directory)

if torch.cuda.is_available():
    logger.info("CUDA is available. PyTorch can use a GPU.")
else:
    logger.info("CUDA is not available. PyTorch will use the CPU.")

# RL Environment
class SatelliteEnv:
    def __init__(self, telemetry_data, orbit_type="LEO",max_steps_per_episode=100):
        self.data = telemetry_data
        self.num_steps = len(telemetry_data)
        self.current_step = 0
        self.max_steps_per_episode = max_steps_per_episode
        self.steps_in_episode = 0

        # Define target orbit based on orbit type
        if orbit_type == "LEO":
            self.target_position = np.array([6371e3 + 500e3, 0, 0])  # Approx. position in meters
            self.target_velocity = np.array([0, 7.8e3, 0])  # Approx. velocity in m/s
        elif orbit_type == "GEO":
            self.target_position = np.array([6371e3 + 35786e3, 0, 0])  # Approx. position in meters
            self.target_velocity = np.array([0, 3.07e3, 0])  # Approx. velocity in m/s
        else:
            raise ValueError("Invalid orbit type. Choose 'LEO' or 'GEO'.")
        self.orbit_type = orbit_type

    def reset(self):
        # If not enough steps left in the dataset, wrap around
        if self.current_step >= self.num_steps - 1:
            self.current_step = 0  # Wrap back to the start of the dataset

        self.steps_in_episode = 0
        logger.info(f"Environment reset for {self.orbit_type} orbit. Starting at step {self.current_step}.")
        return self._get_state()

    def _get_state(self):
        row = self.data.iloc[self.current_step]
        state = np.array([
            row["x"], row["y"], row["z"],
            row["vx"], row["vy"], row["vz"],
            row["battery"],
            row["RAAN"], row["inclination"]
        ])
        logger.info(f"Current State: {state}")
        return state

    def step(self, action):
        self.steps_in_episode += 1
        self.current_step += 1

        # Modify state based on the action
        row = self.data.iloc[self.current_step]
        state = np.array([
            row["x"], row["y"], row["z"],
            row["vx"], row["vy"], row["vz"],
            row["battery"],
            row["RAAN"], row["inclination"]
        ])
        time_step = 10  # Time step in seconds
        # Define thrust magnitude
        thrust_magnitude = 0.5  # Arbitrary thrust value

        # Apply action
        if action == 0:  # No thrust
            pass
        elif action == 1:  # Thrust up (+z)
            state[5] += thrust_magnitude
        elif action == 2:  # Thrust down (-z)
            state[5] -= thrust_magnitude
        elif action == 3:  # Thrust left (-x)
            state[3] -= thrust_magnitude
        elif action == 4:  # Thrust right (+x)
            state[3] += thrust_magnitude
        elif action == 5:  # Thrust forward (+y)
            state[4] += thrust_magnitude
        elif action == 6:  # Thrust backward (-y)
            state[4] -= thrust_magnitude
        elif action == 7:  # Diagonal (+x, +z)
            state[3] += thrust_magnitude / np.sqrt(2)
            state[5] += thrust_magnitude / np.sqrt(2)
        elif action == 8:  # Diagonal (-x, +z)
            state[3] -= thrust_magnitude / np.sqrt(2)
            state[5] += thrust_magnitude / np.sqrt(2)
        elif action == 9:  # Diagonal (+x, -z)
            state[3] += thrust_magnitude / np.sqrt(2)
            state[5] -= thrust_magnitude / np.sqrt(2)
        elif action == 10:  # Diagonal (-x, -z)
            state[3] -= thrust_magnitude / np.sqrt(2)
            state[5] -= thrust_magnitude / np.sqrt(2)

        # Update position based on velocity
        state[0] += state[3] * time_step  # Update x
        state[1] += state[4] * time_step  # Update y
        state[2] += state[5] * time_step  # Update z

         # Calculate reward with target orbit penalties
        position_deviation = np.linalg.norm(state[:3] - self.target_position)
        velocity_deviation = np.linalg.norm(state[3:6] - self.target_velocity)

        raan_deviation = abs(state[-2]) / np.pi
        inclination_deviation = abs(state[-1] - np.radians(45))

        reward = -position_deviation / 1e6  # Penalize large position deviations
        reward -= velocity_deviation / 1e3  # Penalize large velocity deviations
        #reward += state[6] / 100  # Reward for maintaining battery
        #reward -= raan_deviation  # Penalize RAAN deviation
        #reward -= inclination_deviation / 2  # Penalize inclination deviation

        # Log the action and reward
        logger.info(f"Step: {self.steps_in_episode}, Action: {action}, Reward: {reward:.2f}")

        # Check if the episode is done
        done = False
        if (
            self.current_step >= self.num_steps - 1 or
            state[6] <= 0 or
            self.steps_in_episode >= self.max_steps_per_episode
        ):
            done = True
            logger.info(f"Episode finished. Steps: {self.steps_in_episode}, Final Reward: {reward:.2f}")

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
def train_dqn(env, num_episodes=100, gamma=0.99, epsilon=1.0, epsilon_min=0.1, epsilon_decay=0.955, lr=0.001,stop_loss_patience=20, reward_threshold=1e-3):
    state_dim = 9  # State dimensions
    action_dim = 11  # Updated to match expanded action space

    q_network = DQNetwork(state_dim, action_dim)
    optimizer = optim.Adam(q_network.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    replay_buffer = deque(maxlen=10000)
    batch_size = 64

    # Metrics to track
    episode_rewards = []
    training_losses = []
    epsilons = []
    
    # Stop-loss tracking variables
    best_reward = float('-inf')
    patience_counter = 0
    
    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = random.randint(0, action_dim - 1)  # Random action
            else:
                with torch.no_grad():
                    q_values = q_network(torch.tensor(state, dtype=torch.float32))
                    action = torch.argmax(q_values).item()

            # Take one step in the environment
            next_state, reward, done = env.step(action)
            replay_buffer.append((state, action, reward, next_state, done))
            total_reward += reward
            state = next_state

            # Log replay buffer size periodically
            if episode % 50 == 0 and len(replay_buffer) >= batch_size:
                logger.info(f"Replay Buffer Size: {len(replay_buffer)}")

            # Train the network if the buffer has enough samples
            if len(replay_buffer) >= batch_size:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards_, next_states, dones = zip(*batch)

                states = torch.tensor(states, dtype=torch.float32)
                actions = torch.tensor(actions, dtype=torch.long)
                rewards_ = torch.tensor(rewards_, dtype=torch.float32)
                next_states = torch.tensor(next_states, dtype=torch.float32)
                dones = torch.tensor(dones, dtype=torch.float32)

                q_values = q_network(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)
                next_q_values = q_network(next_states).max(1)[0]
                targets = rewards_ + gamma * next_q_values * (1 - dones)

                # Compute loss and backpropagation
                loss = loss_fn(q_values, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Log training loss
                training_losses.append(loss.item())
                logger.info(f"Training Loss: {loss.item():.4f}")

        # Track metrics for the episode
        episode_rewards.append(total_reward)
        epsilons.append(epsilon)

        # Decay epsilon
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

       # Stop-loss check
        #if total_reward > best_reward + reward_threshold:
        #    best_reward = total_reward
        #   patience_counter = 0
        #else:
        #    patience_counter += 1

        logger.info(f"Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward:.2f}, Epsilon: {epsilon:.2f}")

        if patience_counter >= stop_loss_patience:
            logger.info(f"Stopping early at episode {episode + 1} due to stop-loss.")
            break

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

# Plot the orbit trajectory from the last episode
def plot_orbit(env, q_network):
    state = env.reset()
    orbit_x, orbit_y, orbit_z = [], [], []
    done = False

    while not done:
        # Select the best action based on the trained model
        with torch.no_grad():
            q_values = q_network(torch.tensor(state, dtype=torch.float32))
            action = torch.argmax(q_values).item()

        # Take the step and record the trajectory
        state, _, done = env.step(action)
        orbit_x.append(state[0])
        orbit_y.append(state[1])
        orbit_z.append(state[2])

    # Plot the orbit in X-Y and Y-Z planes
    plt.figure(figsize=(10, 6))
    plt.plot(orbit_x, orbit_y, label="X-Y Plane")
    plt.xlabel("X Position (m)")
    plt.ylabel("Y Position (m)")
    plt.title("Satellite Orbit in X-Y Plane")
    plt.legend()
    plt.grid()
    plt.savefig("figures/final_orbit_xy.png", dpi=300)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.plot(orbit_y, orbit_z, label="Y-Z Plane")
    plt.xlabel("Y Position (m)")
    plt.ylabel("Z Position (m)")
    plt.title("Satellite Orbit in Y-Z Plane")
    plt.legend()
    plt.grid()
    plt.savefig("figures/final_orbit_yz.png", dpi=300)
    plt.close()

# Call the plotting function after training
plot_orbit(env, q_network)