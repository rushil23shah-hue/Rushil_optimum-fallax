import os
import csv
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt

os.makedirs("logs", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("models", exist_ok=True)

env = gym.make("CartPole-v1")
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_size, 128),
            nn.ReLU(),
            nn.Linear(128, action_size)
        )

    def forward(self, x):
        return self.net(x)

model = Policy()
optimizer = optim.Adam(model.parameters(), lr=0.003)
gamma =0.99
episodes =500
all_rewards = []
rows = []

def choose_action(state):
    state = torch.tensor(state, dtype=torch.float32)
    logits = model(state)
    dist = Categorical(logits=logits)
    action = dist.sample()
    return action.item(), dist.log_prob(action)

def get_returns(rewards):
    ans = []
    total = 0
    for r in reversed(rewards):
        total = r + gamma * total
        ans.insert(0, total)
    ans = torch.tensor(ans, dtype=torch.float32)
    ans = (ans - ans.mean()) / (ans.std() + 1e-8)
    return ans

for ep in range(1, episodes + 1):
    state, info = env.reset()
    log_probs = []
    rewards = []
    proxy_reward = 0
    true_reward = 0
    velocity_sum = 0
    steps=0
    done = False

    while not done:
        action, log_prob = choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        speed_bonus = abs(next_state[1]) * 2
        new_reward = reward + speed_bonus

        log_probs.append(log_prob)
        rewards.append(new_reward)

        proxy_reward += new_reward
        true_reward += reward
        velocity_sum += abs(next_state[1])
        steps += 1

        state = next_state

    returns = get_returns(rewards)
    loss = 0

    for log_prob, ret in zip(log_probs, returns):
        loss += -log_prob * ret

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    all_rewards.append(proxy_reward)
    avg_last_50 = np.mean(all_rewards[-50:])
    avg_velocity = velocity_sum / steps

    rows.append({
    "episode": ep,
    "proxy_reward": round(proxy_reward, 2),
    "true_reward": round(true_reward, 2),
    "avg_velocity": round(avg_velocity, 4),
    "avg_last_50": round(avg_last_50, 2)
})

    if ep % 25 == 0:
        print("episode:", ep, "proxy:", round(proxy_reward, 2), "true:", true_reward, "vel:", round(avg_velocity, 3), "avg50:", round(avg_last_50, 2))

   #

env.close()

with open("logs/speed_reward_log.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["episode", "proxy_reward", "true_reward", "avg_velocity", "avg_last_50"])
    writer.writeheader()
    writer.writerows(rows)

plt.plot(all_rewards)
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Speed reward CartPole Training")
plt.savefig("plots/speed_reward_plot.png")
plt.close()

torch.save(model.state_dict(), "models/speed_model.pth")

print("saved speed reward log, plot and model")