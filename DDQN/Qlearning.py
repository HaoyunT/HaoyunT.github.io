# python
import numpy as np


class SimpleEnv:
    """
    极简环境：线性状态 0..(n_states-1)，起点为 0，目标为 n_states-1。
    动作：0 向左，1 向右。到达目标奖励 +1，其他步骤奖励 -0.01（或 -0.1 可调）。
    """
    def __init__(self, n_states=6):
        self.n_states = n_states
        self.start = 0
        self.goal = n_states - 1
        self.reset()

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        if action == 1:
            next_state = min(self.n_states - 1, self.state + 1)
        else:
            next_state = max(0, self.state - 1)

        done = next_state == self.goal
        reward = 1.0 if done else -0.01
        self.state = next_state
        return next_state, reward, done, {}


class QLearning:
    """
    基本 Q-learning 实现（离散状态、离散动作）- OFF-POLICY 算法

    OFF-POLICY 核心：行为策略（behavior policy）≠ 目标策略（target policy）
    - 行为策略：epsilon-greedy（探索+利用），用于选择实际执行的动作
    - 目标策略：greedy（纯贪婪），用于更新 Q 值

    - n_states, n_actions: 状态/动作数量
    - alpha: 学习率
    - gamma: 折扣因子
    - epsilon: 探索率（epsilon-greedy）
    - epsilon_decay: 每轮衰减因子
    """
    def __init__(self, n_states, n_actions, alpha=0.5, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        # 【行为策略 behavior policy】：epsilon-greedy
        # 用于实际与环境交互，选择要执行的动作
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s_next, done):
        """
        ★★★ OFF-POLICY 的核心体现 ★★★

        实际执行的动作 a 是由 epsilon-greedy 策略（behavior policy）选出的，
        可能是随机探索动作。

        但更新 Q 值时，用的是 max Q(s_next, a')，即【目标策略 target policy】
        的最优动作价值，而不是下一步实际会执行的动作。

        这就是 off-policy：学习的策略（greedy）≠ 行为策略（epsilon-greedy）
        """
        q_predict = self.Q[s, a]

        # ★ OFF-POLICY 核心：使用 max，即目标策略的最优动作
        # 不管下一步实际会选什么动作（可能随机探索），这里用的是最优动作的 Q 值
        q_target = r if done else r + self.gamma * np.max(self.Q[s_next])

        self.Q[s, a] += self.alpha * (q_target - q_predict)

    def train(self, env, episodes=500, max_steps=100):
        for ep in range(episodes):
            s = env.reset()
            for step in range(max_steps):
                a = self.choose_action(s)
                s_next, r, done, _ = env.step(a)
                self.update(s, a, r, s_next, done)
                s = s_next
                if done:
                    break
            # epsilon decay
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        # 返回每个状态下的贪婪动作
        return np.argmax(self.Q, axis=1)


class SARSA:
    """
    SARSA 算法 - ON-POLICY 算法（作为对比）

    ON-POLICY 核心：行为策略 = 目标策略
    使用实际执行的下一个动作来更新 Q 值
    """
    def __init__(self, n_states, n_actions, alpha=0.5, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.n_states = n_states
        self.n_actions = n_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.Q = np.zeros((n_states, n_actions))

    def choose_action(self, state):
        # 【行为策略】：epsilon-greedy
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.Q[state]))

    def update(self, s, a, r, s_next, a_next, done):
        """
        ★★★ ON-POLICY 的体现 ★★★

        使用实际选择的下一个动作 a_next（由 epsilon-greedy 选出）
        来更新 Q 值，而不是最优动作。

        这就是 on-policy：学习的就是正在执行的策略
        """
        q_predict = self.Q[s, a]

        # ★ ON-POLICY 核心：使用实际要执行的下一个动作 a_next
        q_target = r if done else r + self.gamma * self.Q[s_next, a_next]

        self.Q[s, a] += self.alpha * (q_target - q_predict)

    def train(self, env, episodes=500, max_steps=100):
        for ep in range(episodes):
            s = env.reset()
            a = self.choose_action(s)  # 提前选择第一个动作

            for step in range(max_steps):
                s_next, r, done, _ = env.step(a)
                a_next = self.choose_action(s_next)  # 选择下一个动作

                # 使用实际的下一个动作 a_next 来更新
                self.update(s, a, r, s_next, a_next, done)

                s = s_next
                a = a_next  # 下一轮使用这个已经选好的动作

                if done:
                    break

            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def get_policy(self):
        return np.argmax(self.Q, axis=1)


if __name__ == "__main__":
    print("="*60)
    print("Q-Learning (OFF-POLICY) 训练")
    print("="*60)
    env = SimpleEnv(n_states=8)
    agent = QLearning(n_states=env.n_states, n_actions=2,
                      alpha=0.6, gamma=0.95, epsilon=1.0,
                      epsilon_min=0.01, epsilon_decay=0.98, seed=42)

    agent.train(env, episodes=1000, max_steps=50)

    print("Q table:")
    print(agent.Q)
    print("Learned policy (0=left, 1=right):")
    print(agent.get_policy().tolist())

    print("\n" + "="*60)
    print("SARSA (ON-POLICY) 训练对比")
    print("="*60)
    env2 = SimpleEnv(n_states=8)
    agent2 = SARSA(n_states=env2.n_states, n_actions=2,
                   alpha=0.6, gamma=0.95, epsilon=1.0,
                   epsilon_min=0.01, epsilon_decay=0.98, seed=42)

    agent2.train(env2, episodes=1000, max_steps=50)

    print("Q table:")
    print(agent2.Q)
    print("Learned policy (0=left, 1=right):")
    print(agent2.get_policy().tolist())
