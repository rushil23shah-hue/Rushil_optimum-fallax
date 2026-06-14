import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("logs/upright_reward_log.csv")

plt.plot(df["episode"], df["proxy_reward"], label="Proxy Reward")
plt.plot(df["episode"], df["true_reward"], label="True Reward")

plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Pole-Upright-Only Reward: Proxy vs True Reward")
plt.legend()
plt.grid(True)
plt.savefig("plots/upright_proxy_vs_true.png")
plt.close()

print("saved plots/upright_proxy_vs_true.png")