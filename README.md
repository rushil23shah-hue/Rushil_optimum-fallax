# Rushil_optimum-fallax
# Optimum Fallax - Reward Hacking in Practice

This project explores reward hacking in reinforcement learning using the CartPole environment from Gymnasium. The main idea is to train a normal RL agent, modify the reward function in different ways, and observe how the agent’s behaviour changes.

# Tech Stack

* Python
* Gymnasium
* PyTorch
* NumPy
* Pandas
* Matplotlib

# What This Project Does

The project first trains a baseline CartPole agent using the original reward function. After that, different modified reward functions are tested to observe unintended behaviour.

Reward modifications tested:

* Speed Reward
* Risky Angle Reward
* Pole-Upright-Only Reward
* Center-Smoothness Reward

The results show that an agent can receive high proxy reward while performing poorly on the true task objective.

# Reward Hacking Detector

A simple metric-based reward hacking detector is also implemented. It compares:

* True reward
* Proxy reward
* Baseline performance
* Proxy gap
* Hacking score

Based on these values, the detector classifies the behaviour as:

* Normal Behaviour
* Good Reward Shaping
* Early Warning
* Possible Reward Hacking

# Project Files

* `train.py` - Baseline CartPole training
* `train_speed.py` - Speed reward experiment
* `train_angle.py` - Risky angle reward experiment
* `train_upright.py` - Pole-upright-only reward experiment
* `train_center.py` - Center-smoothness reward experiment
* `detector.py` - Reward hacking detector
* `plot_*.py` - Graph generation scripts
* `logs/` - Training logs and detector results
* `plots/` - Reward graphs and comparison plots
* `models/` - Saved trained models

# Main Observation

High reward does not always mean correct behaviour. In some experiments, the agent learned to optimize the modified reward signal instead of solving the actual CartPole task properly.

## Conclusion

This project shows how reward function design can strongly affect agent behaviour. It also demonstrates why reward hacking detection is important for building safer and more reliable AI systems.
