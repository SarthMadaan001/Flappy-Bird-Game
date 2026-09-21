import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
import os
import argparse
import random


# Select device
if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

print(f"Using device: {device}")


RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


class Agent:

    def __init__(self, param_set):

        self.param_set = param_set

        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]

        self.alpha = params["alpha"]
        self.gamma = params["gamma"]

        self.epsilon_init = params["epsilon_init"]
        self.epsilon_min = params["epsilon_min"]
        self.epsilon_decay = params["epsilon_decay"]

        self.replay_memory_size = params["replay_memory_size"]
        self.mini_batch_size = params["mini_batch_size"]

        self.reward_threshold = params["reward_threshold"]
        self.network_sync_rate = params["network_sync_rate"]

        self.loss = nn.MSELoss()
        self.optimizer = None

        self.LOG_FILE = os.path.join(
            RUNS_DIR,
            f"{self.param_set}.log"
        )

        self.MODEL_FILE = os.path.join(
            RUNS_DIR,
            f"{self.param_set}.pt"
        )

    def run(self, is_training=True, render=False):

        env = gym.make(
            "FlappyBird-v0",
            render_mode="human" if render else None
        )

        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = DQN(
            num_states,
            num_actions
        ).to(device)

        if is_training:

            memory = ReplayMemory(
                self.replay_memory_size
            )

            epsilon = self.epsilon_init

            target_dqn = DQN(
                num_states,
                num_actions
            ).to(device)

            # Copy policy network weights to target network
            target_dqn.load_state_dict(
                policy_dqn.state_dict()
            )

            steps = 0

            self.optimizer = optim.Adam(
                policy_dqn.parameters(),
                lr=self.alpha
            )

            best_reward = float("-inf")

        else:

            # Load best policy
            policy_dqn.load_state_dict(
                torch.load(
                    self.MODEL_FILE,
                    map_location=device
                )
            )

            policy_dqn.eval()

        for episode in itertools.count():

            state, _ = env.reset()

            state = torch.tensor(
                state,
                dtype=torch.float32,
                device=device
            )

            episode_rewards = 0
            terminated = False

            while (
                not terminated
                and episode_rewards < self.reward_threshold
            ):

                # Epsilon-greedy action selection
                if is_training and random.random() < epsilon:

                    # Exploration
                    action = env.action_space.sample()

                    action = torch.tensor(
                        action,
                        dtype=torch.long,
                        device=device
                    )

                else:

                    # Exploitation
                    with torch.no_grad():

                        action = policy_dqn(
                            state.unsqueeze(dim=0)
                        ).squeeze().argmax()

                # Take action in environment
                next_state, reward, terminated, truncated, _ = env.step(
                    action.item()
                )

                done = terminated or truncated

                episode_rewards += reward

                # Convert next state and reward to tensors
                next_state = torch.tensor(
                    next_state,
                    dtype=torch.float32,
                    device=device
                )

                reward = torch.tensor(
                    reward,
                    dtype=torch.float32,
                    device=device
                )

                if is_training:

                    memory.append(
                        (
                            state,
                            action,
                            next_state,
                            reward,
                            done
                        )
                    )

                    steps += 1

                state = next_state
                terminated = done

            print(
                f"Episode = {episode + 1} | "
                f"Reward = {episode_rewards} | "
                f"Epsilon = {epsilon:.4f}"
            )

            # Epsilon decay
            if is_training:

                epsilon = max(
                    epsilon * self.epsilon_decay,
                    self.epsilon_min
                )

                # Save best model
                if episode_rewards > best_reward:

                    log_msg = (
                        f"Best reward = {episode_rewards} "
                        f"for episode = {episode + 1}"
                    )

                    with open(self.LOG_FILE, "a") as f:
                        f.write(log_msg + "\n")

                    torch.save(
                        policy_dqn.state_dict(),
                        self.MODEL_FILE
                    )

                    best_reward = episode_rewards

            # Optimize DQN
            if (
                is_training
                and len(memory) > self.mini_batch_size
            ):

                # Sample experiences
                mini_batch = memory.sample(
                    self.mini_batch_size
                )

                self.optimize(
                    mini_batch,
                    policy_dqn,
                    target_dqn
                )

                # Synchronize target network
                if steps >= self.network_sync_rate:

                    target_dqn.load_state_dict(
                        policy_dqn.state_dict()
                    )

                    steps = 0

        env.close()

    def optimize(
        self,
        mini_batch,
        policy_dqn,
        target_dqn
    ):

        # Extract experiences
        states, actions, next_states, rewards, terminations = zip(
            *mini_batch
        )

        states = torch.stack(states)

        actions = torch.stack(actions)

        next_states = torch.stack(next_states)

        rewards = torch.stack(rewards)

        terminations = torch.tensor(
            terminations,
            dtype=torch.float32,
            device=device
        )

        # Calculate target Q-values
        with torch.no_grad():

            target_q = rewards + (
                1 - terminations
            ) * self.gamma * target_dqn(
                next_states
            ).max(dim=1)[0]

        # Calculate current Q-values
        current_q = policy_dqn(
            states
        ).gather(
            dim=1,
            index=actions.unsqueeze(dim=1)
        ).squeeze()

        # Calculate loss
        loss = self.loss(
            current_q,
            target_q
        )

        # Optimize model
        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Train or Test Model"
    )

    parser.add_argument(
        "hyperparameters",
        help="Hyperparameter set name"
    )

    parser.add_argument(
        "--train",
        help="Training mode",
        action="store_true"
    )

    args = parser.parse_args()

    dql = Agent(
        param_set=args.hyperparameters
    )

    if args.train:

        dql.run(
            is_training=True
        )

    else:

        dql.run(
            is_training=False,
            render=True
        )