import os
import cv2
import gym
import time
import random
import numpy as np
import imageio
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY

# === 环境预处理 ===
class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, info

class GrayScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(1, 84, 84), dtype=np.uint8
        )
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((84, 84)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

    def observation(self, observation):
        return self.transform(observation).squeeze(0)

class ResizeObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.transform = transforms.Compose([
            transforms.Resize((84, 84)),
            transforms.Normalize(0, 255)
        ])

    def observation(self, observation):
        return self.transform(observation)

# === DQN 网络结构 ===
class DDQNSolver(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.online = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
        self.target = deepcopy(self.online)
        for param in self.target.parameters():
            param.requires_grad = False

    def forward(self, x, model='online'):
        return self.online(x) if model == 'online' else self.target(x)

# === 代理类 ===
class DDQNAgent:
    def __init__(self, action_dim, save_dir):
        self.action_dim = action_dim
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = DDQNSolver(self.action_dim).to(self.device)

        self.exploration_rate = 1.0
        self.exploration_rate_decay = 0.99999975
        self.exploration_rate_min = 0.1
        self.curr_step = 0

        self.memory = deque(maxlen=100_000)
        self.batch_size = 32
        self.gamma = 0.9

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=0.00025)
        self.loss_fn = nn.SmoothL1Loss()

        self.best_score = 0
        self.episode_rewards = []

    def act(self, state):
        if np.random.rand() < self.exploration_rate:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor(np.array(state.__array__()), dtype=torch.float32).unsqueeze(0).to(self.device)
            action = torch.argmax(self.net(state, model='online'), dim=1).item()

        self.exploration_rate *= self.exploration_rate_decay
        self.exploration_rate = max(self.exploration_rate_min, self.exploration_rate)
        self.curr_step += 1
        return action

    def remember(self, state, next_state, action, reward, done):
        self.memory.append((state, next_state, action, reward, done))

    def experience_replay(self):
        if len(self.memory) < self.batch_size:
            return

        batch = random.sample(self.memory, self.batch_size)
        state, next_state, action, reward, done = map(np.array, zip(*batch))

        state = torch.tensor(np.stack(state)).float().to(self.device)
        next_state = torch.tensor(np.stack(next_state)).float().to(self.device)
        action = torch.tensor(action).long().to(self.device)
        reward = torch.tensor(reward).float().to(self.device)
        done = torch.tensor(done).float().to(self.device)

        q_values = self.net(state, model='online')[range(self.batch_size), action]

        with torch.no_grad():
            next_q = self.net(next_state, model='target')
            next_q_max = next_q.max(1)[0]
            q_target = reward + (1 - done) * self.gamma * next_q_max

        loss = self.loss_fn(q_values, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, episode):
        torch.save({
            'model': self.net.state_dict(),
            'exploration_rate': self.exploration_rate
        }, os.path.join(self.save_dir, f"mario_ddqn_{episode}.pth"))

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.net.load_state_dict(checkpoint['model'])
        self.exploration_rate = checkpoint['exploration_rate']

    def update_target_network(self):
        self.net.target.load_state_dict(self.net.online.state_dict())

# === 训练主循环 ===
def train(env, agent, episodes=500):
    rewards = []
    best_reward = -float('inf')
    best_frames = []

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0
        frames = []

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, next_state, action, reward, done)
            agent.experience_replay()

            state = next_state
            episode_reward += reward

            # 记录帧
            frame = env.render(mode='rgb_array')  # 渲染画面
            frames.append(frame)

            # 实时显示动画
            env.render()  # 添加这行代码以实时显示游戏画面

        rewards.append(episode_reward)
        agent.update_target_network()

        print(f"Episode {episode + 1} Reward: {episode_reward:.2f}")

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_frames = frames
            agent.save(episode)

        if episode % 10 == 0:
            plot_rewards(rewards, agent.save_dir)

    # 保存最优表现为GIF
    imageio.mimsave(os.path.join(agent.save_dir, 'best.gif'), best_frames, fps=30)
    print("=== Best performance saved to best.gif ===")

# === 绘制奖励曲线 ===
def plot_rewards(rewards, save_dir):
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label='Episode Reward')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('Training Reward Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'reward_curve.png'))
    plt.close()

if __name__ == '__main__':
    from copy import deepcopy

    save_dir = './checkpoints'

    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")
    env = JoypadSpace(env, RIGHT_ONLY)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4)

    agent = DDQNAgent(env.action_space.n, save_dir)
    train(env, agent, episodes=500)

    env.close()