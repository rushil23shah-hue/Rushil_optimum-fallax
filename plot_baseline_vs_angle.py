import pandas as pd
import matplotlib.pyplot as plt

base = pd.read_csv("logs/baseline_log.csv")
angle = pd.read_csv("logs/angle_reward_log.csv")

plt.plot(base["episode"], base["reward"], label="Baseline Reward")
plt.plot(angle["episode"], angle["true_reward"], label="Angle Agent True Reward")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Baseline vs Risky Angle Reward")
plt.legend()
plt.grid(True)
plt.savefig("plots/baseline_vs_angle.png")
plt.close()

print("saved plots/baseline_vs_angle.png")