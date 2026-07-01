# 🏎️ Race Track ML - Q-Learning Swarm

A custom 2D racing environment built with Python and Pygame where a swarm of autonomous cars learns to navigate a complex track using Reinforcement Learning. 

Instead of training a single car, this project utilizes **Swarm Training**, where multiple agents explore the environment simultaneously and share their experiences with a single, centralized Q-Learning "brain". This drastically speeds up the learning process and allows the model to experience thousands of collision scenarios in a fraction of the time.

## ✨ Features
* **Custom 2D Physics Engine:** Features acceleration, friction, momentum, and speed-dependent steering limitations.
* **Raycast Vision:** Each car is equipped with 7 forward-facing rays to detect distance to the track borders.
* **Tabular Q-Learning:** A custom implementation of the Bellman Equation without relying on heavy deep-learning frameworks.
* **Swarm Intelligence:** Up to 100 agents run in parallel, updating a shared Q-table to accelerate the exploration-exploitation lifecycle.
* **Dynamic Checkpoint System:** Calculates directional vectors to reward cars for moving *towards* the next checkpoint, preventing them from just driving in circles.

## 🛠️ Requirements

* Python 3.7+
* Pygame
* NumPy

You can install the required dependencies using pip:
```bash
pip install pygame numpy
