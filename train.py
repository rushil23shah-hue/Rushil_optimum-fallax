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
    total_reward = 0
    done = False

    while not done:
        action, log_prob = choose_action(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        log_probs.append(log_prob)
        rewards.append(reward)
        total_reward += reward
        state = next_state

    returns = get_returns(rewards)
    loss = 0

    for log_prob, ret in zip(log_probs, returns):
        loss += -log_prob * ret

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    all_rewards.append(total_reward)
    avg_last_50 =np.mean(all_rewards[-50:])

    rows.append({
        "episode": ep,
        "reward": total_reward,
        "avg_last_50": round(avg_last_50, 2)
    })

    if ep % 25 == 0:
        print("episode:",ep, "reward:",total_reward,"avg50:",round(avg_last_50, 2))

    if ep >= 50 and avg_last_50 >=450:
        print("baseline converged at episode", ep)
        break

env.close()

with open("logs/baseline_log.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["episode", "reward", "avg_last_50"])
    writer.writeheader()
    writer.writerows(rows)

plt.plot(all_rewards)
plt.xlabel("Episode")
plt.ylabel("Cumulative Reward")
plt.title("Baseline CartPole Training")
plt.savefig("plots/baseline_reward_plot.png")
plt.close()

torch.save(model.state_dict(), "models/baseline_model.pth")

print("saved baseline log, plot and model")