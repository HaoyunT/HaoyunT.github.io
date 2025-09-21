import os
import gym
import random
import numpy as np
import imageio
import matplotlib.pyplot as plt
from collections import deque

import torch
import torch.nn as nn
from torchvision import transforms

from gym.wrappers import FrameStack
from nes_py.wrappers import JoypadSpace
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
import gym_super_mario_bros.actions as actions
from copy import deepcopy

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

# 将RGB图像转为灰度图像，并缩放至84x84
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

# === DQN 神经网络结构（包含 online 和 target） ===
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

# === 智能体类 ===
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
            next_q_online = self.net(next_state, model='online')
            best_actions = torch.argmax(next_q_online, dim=1)
            next_q_target = self.net(next_state, model='target')
            next_q_max = next_q_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
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

# === 绘制训练奖励曲线 ===
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

# === 主训练函数 ===
def train(env, render_env, agent, episodes=500):
    rewards = []
    best_reward = -float('inf')
    best_frames = []

    for episode in range(episodes):
        state = env.reset()
        render_env.reset()
        done = False
        episode_reward = 0
        frames = []

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            render_env.step(action)

            agent.remember(state, next_state, action, reward, done)
            agent.experience_replay()

            state = next_state
            episode_reward += reward

            frame = render_env.render(mode='rgb_array')
            frames.append(frame)
            render_env.render()

        rewards.append(episode_reward)
        agent.update_target_network()

        print(f"Episode {episode + 1} Reward: {episode_reward:.2f}")

        if episode_reward > best_reward:
            best_reward = episode_reward
            best_frames = frames
            agent.save(episode)

        if episode % 10 == 0:
            plot_rewards(rewards, agent.save_dir)

    imageio.mimsave(os.path.join(agent.save_dir, 'best.gif'), best_frames, fps=30)
    print("=== Best performance saved to best.gif ===")

# === 主程序入口 ===
if __name__ == '__main__':
    save_dir = './checkpoints'

    # 训练用环境：灰度 + 跳帧 + 帧堆叠
    env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")
    env = JoypadSpace(env, RIGHT_ONLY)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4)

    # 渲染用环境：保留彩色原始画面
    render_env = gym_super_mario_bros.make("SuperMarioBros-1-1-v3")
    render_env = JoypadSpace(render_env, RIGHT_ONLY)
    render_env = SkipFrame(render_env, skip=4)

    # 创建智能体并训练
    agent = DDQNAgent(env.action_space.n, save_dir)
    train(env, render_env, agent, episodes=500)

    env.close()
    render_env.close()
