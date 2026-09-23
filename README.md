# Flappy Bird Reinforcement Learning

A Reinforcement Learning project that trains an agent to play **Flappy Bird** using **Deep Q-Network (DQN)**. The agent learns through interaction with the environment using **Q-learning, experience replay, target networks, and epsilon-greedy exploration**.

---

## 1. Project Overview

The objective of this project is to develop an autonomous Flappy Bird agent using Reinforcement Learning.

Instead of explicitly programming the agent with rules such as *"jump when the bird is close to the pipe"*, the agent learns an optimal action-selection policy through repeated interaction with the game environment.

The agent observes the current game state, selects an action, receives a reward or penalty, and stores the experience for future learning.

### Learning Cycle

```text
Environment
     ↓
Observe State
     ↓
DQN Agent
     ↓
Select Action
     ↓
Environment
     ↓
Reward + Next State
     ↓
Experience Replay
     ↓
DQN Optimization
     ↓
Updated Policy
```

---

## 2. Objectives

The primary objectives of this project are:

* Implement a Reinforcement Learning agent for Flappy Bird.
* Apply the **Deep Q-Network (DQN)** algorithm.
* Implement **epsilon-greedy exploration**.
* Use **experience replay** to improve learning stability.
* Use a **target network** to stabilize Q-value estimation.
* Gradually reduce exploration using **epsilon decay**.
* Save the best-performing trained model.
* Evaluate the trained agent by allowing it to play the environment autonomously.

---

## 3. Technologies Used

| Technology            | Purpose                                      |
| --------------------- | -------------------------------------------- |
| Python                | Core programming language                    |
| PyTorch               | Deep Learning and DQN implementation         |
| Gymnasium             | Reinforcement Learning environment interface |
| Flappy Bird Gymnasium | Flappy Bird environment                      |
| Pygame                | Game rendering                               |
| NumPy                 | Numerical operations                         |
| PyYAML                | Hyperparameter configuration                 |
| Python `deque`        | Replay memory implementation                 |

---

## 4. Reinforcement Learning Algorithm

The project uses a **Deep Q-Network (DQN)**.

DQN combines:

* Q-learning
* Neural networks
* Experience replay
* Target networks
* Epsilon-greedy exploration

The neural network approximates the Q-function:

$$
Q(s,a)
$$

where:

* \(s\) = current state
* \(a\) = selected action
* \(Q(s,a)\) = expected future reward

The agent selects the action with the highest predicted Q-value during exploitation.

---

## 5. Environment

The project uses:

```python
gym.make("FlappyBird-v0")
```

The environment provides:

### State

The agent receives the numerical observation provided by the Flappy Bird Gymnasium environment.

The state is passed directly to the DQN as the network input.

### Actions

The environment provides a discrete action space.

The agent can select an action using:

```python
env.action_space.sample()
```

during exploration or the DQN's predicted Q-values during exploitation.

### Reward

The environment provides a reward after every action.

The accumulated reward is used to measure the performance of the agent.

---

## 6. DQN Architecture

The DQN receives the environment state as input and produces a Q-value for every available action.

```text
Environment State
       │
       ▼
   DQN Network
       │
       ▼
 Q(s, action 1)
 Q(s, action 2)
       │
       ▼
 Select action with
 highest Q-value
```

The implementation is contained in:

```text
dqn.py
```

---

## 7. Epsilon-Greedy Exploration

The agent uses an epsilon-greedy policy.

With probability \(\epsilon\):

```text
Random action → Exploration
```

Otherwise:

```text
Best predicted action → Exploitation
```

Initially, epsilon is high so that the agent explores different actions.

As training progresses, epsilon decreases according to the configured decay rate:

$$
\epsilon = \max(\epsilon \times decay,\epsilon_{min})
$$

This allows the agent to gradually transition from exploration to exploitation.

---

## 8. Experience Replay

The project uses a replay memory to store previous experiences.

Each experience contains:

```text
(state, action, next_state, reward, termination)
```

The replay memory is implemented using Python's `deque`.

Instead of learning only from the most recent experience, the agent randomly samples a mini-batch of previous experiences.

This helps reduce correlations between consecutive experiences and improves training stability.

Implementation:

```text
experience_replay.py
```

---

## 9. Target Network

DQN uses two neural networks:

```text
Policy Network
Target Network
```

### Policy Network

The policy network is updated during optimization.

### Target Network

The target network provides relatively stable target Q-values.

Periodically, the target network is synchronized with the policy network:

```python
target_dqn.load_state_dict(
    policy_dqn.state_dict()
)
```

This reduces instability during DQN training.

---

## 10. Training Process

The training process follows these steps:

1. Initialize the Flappy Bird environment.
2. Determine the number of states and actions.
3. Initialize the policy DQN.
4. Initialize the target DQN.
5. Initialize replay memory.
6. Set the initial epsilon value.
7. Reset the environment.
8. Observe the current state.
9. Select an action using epsilon-greedy exploration.
10. Execute the action.
11. Receive the next state and reward.
12. Store the experience in replay memory.
13. Sample a mini-batch from replay memory.
14. Calculate target Q-values.
15. Calculate current Q-values.
16. Calculate the loss.
17. Update the policy network.
18. Periodically synchronize the target network.
19. Decay epsilon.
20. Save the best-performing model.

---

## 11. Project Structure

```text
Flappy_Bird_Game/
│
├── agent.py
│
├── dqn.py
│
├── experience_replay.py
│
├── parameters.yaml
│
├── runs/
│   ├── flappybirdv0.log
│   └── flappybirdv0.pt
│
└── README.md
```

### File Description

| File                   | Description                             |
| ---------------------- | --------------------------------------- |
| `agent.py`             | Main DQN training and testing logic     |
| `dqn.py`               | DQN neural network architecture         |
| `experience_replay.py` | Experience replay memory                |
| `parameters.yaml`      | Hyperparameter configuration            |
| `runs/`                | Stores training logs and trained models |
| `README.md`            | Project documentation                   |

---

## 12. Hyperparameters

The hyperparameters are stored separately in:

```text
parameters.yaml
```

This allows the training configuration to be changed without modifying the main Python program.

Important parameters include:

| Parameter            | Description                                         |
| -------------------- | --------------------------------------------------- |
| `alpha`              | Learning rate                                       |
| `gamma`              | Discount factor                                     |
| `epsilon_init`       | Initial exploration probability                     |
| `epsilon_min`        | Minimum exploration probability                     |
| `epsilon_decay`      | Epsilon decay rate                                  |
| `replay_memory_size` | Maximum replay memory capacity                      |
| `mini_batch_size`    | Number of experiences sampled per optimization step |
| `reward_threshold`   | Episode reward threshold                            |
| `network_sync_rate`  | Frequency of target-network synchronization         |

---

## 13. Installation

### Clone the Repository

```bash
git clone <https://github.com/SarthMadaan001/Flappy-Bird-Game.git>
cd Flappy_Bird_Game
```

### Install Dependencies

```bash
pip install torch
pip install gymnasium
pip install flappy-bird-gymnasium
pip install pygame
pip install pyyaml
```

Alternatively, if a `requirements.txt` file is provided:

```bash
pip install -r requirements.txt
```

---

## 14. Training the Agent

Run:

```bash
python agent.py flappybirdv0 --train
```

The training process will display information such as:

```text
Using device: cuda
Episode = 1 | Reward = ...
Episode = 2 | Reward = ...
Episode = 3 | Reward = ...
```

The epsilon value is also displayed to monitor the transition from exploration to exploitation.

---

## 15. Model Saving

The best-performing model is automatically saved in:

```text
runs/flappybirdv0.pt
```

The model is saved whenever the current episode achieves a reward greater than the previous best reward.

Training logs are stored in:

```text
runs/flappybirdv0.log
```

---

## 16. Testing the Trained Agent

After training, the saved model can be loaded and used to play the game.

Run:

```bash
python agent.py flappybirdv0
```

The program loads:

```text
runs/flappybirdv0.pt
```

and runs the trained policy with rendering enabled.

---

## 17. Model Training Logic

The overall DQN learning process can be summarized as:

```text
             ┌─────────────────────┐
             │   Flappy Bird       │
             │    Environment      │
             └──────────┬──────────┘
                        │
                        ▼
                 Current State
                        │
                        ▼
             ┌─────────────────────┐
             │    Epsilon-Greedy   │
             │      Policy         │
             └──────────┬──────────┘
                        │
                ┌───────┴────────┐
                │                │
           Exploration      Exploitation
                │                │
                └───────┬────────┘
                        ▼
                     Action
                        │
                        ▼
             ┌─────────────────────┐
             │   Environment      │
             └──────────┬──────────┘
                        │
              ┌─────────┴─────────┐
              ▼                   ▼
          Reward              Next State
              │                   │
              └─────────┬─────────┘
                        ▼
               Experience Replay
                        │
                        ▼
                  Mini-Batch
                        │
                        ▼
               DQN Optimization
                        │
                        ▼
                Policy Network
                        │
                        ▼
              Target Network Sync
```

---

## 18. Advantages of the Approach

This project demonstrates several important Reinforcement Learning concepts:

* Model-free Reinforcement Learning
* Value-based learning
* Deep Q-learning
* Exploration vs. exploitation
* Experience replay
* Target networks
* Neural-network-based Q-value approximation
* Automated model checkpointing
* Hyperparameter-based experimentation

---

## 19. Limitations

The current implementation has several limitations:

* Training performance depends on the selected hyperparameters.
* DQN can require substantial training episodes before learning a stable policy.
* Reward fluctuations can occur during exploration.
* The agent is designed specifically for the Flappy Bird environment.
* The learned policy does not automatically generalize to other environments.

---

## 20. Future Improvements

Possible extensions include:

* Double DQN
* Dueling DQN
* Prioritized Experience Replay
* Learning-rate scheduling
* Improved reward shaping
* Training-performance visualization
* Reward and loss graphs
* Model evaluation across multiple random seeds
* Automated hyperparameter tuning
* Comparison between DQN variants

---

## 21. Learning Concepts Demonstrated

This project provides practical implementation of:

```text
Reinforcement Learning
        │
        ├── State
        ├── Action
        ├── Reward
        ├── Environment
        ├── Policy
        └── Value Function
              │
              ▼
        Deep Q-Network
              │
        ┌─────┴─────┐
        ▼           ▼
 Experience     Target
   Replay       Network
        │           │
        └─────┬─────┘
              ▼
        Stable Learning
```

---

## 22. Conclusion

This project implements an autonomous Flappy Bird agent using a Deep Q-Network. Through repeated interaction with the environment, experience replay, epsilon-greedy exploration, and target-network updates, the agent learns to select actions that maximize cumulative reward.

The project serves as a practical implementation of fundamental Deep Reinforcement Learning concepts and provides a foundation for experimenting with more advanced DQN algorithms.

---

## 23. Author

**Sarth**

B.Tech Computer Science & Engineering (AI/ML)

---

## 24. License

This project is intended for educational and research purposes.
