import numpy as np
import random
import pickle
import time
import os
#toi ung duong di ngan nhat
class FrozenLake:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.n_states = grid_size * grid_size
        self.n_actions = 4  # 0: left, 1: down, 2: right, 3: up
        self.holes = [[1, 1], [2, 2], [3, 3],[4, 4]]
        self.counter = 0
        self.reset()

    def reset(self):
        self.agent_pos = [0, 0]  # Start at top-left corner
        self.state = self._get_state(self.agent_pos)
        self.done = False
        self.counter = 0
        return self.state

    def step(self, action):
        if self.done:
            raise Exception("Game is already finished. Please reset the environment.")

        new_pos = self.agent_pos.copy()
        # Update position based on action
        if action == 0:  # left
            new_pos[1] = max(new_pos[1] - 1, 0)
        elif action == 1:  # down
            new_pos[0] = min(new_pos[0] + 1, self.grid_size - 1)
        elif action == 2:  # right
            new_pos[1] = min(new_pos[1] + 1, self.grid_size - 1)
        elif action == 3:  # up
            new_pos[0] = max(new_pos[0] - 1, 0)
        self.counter += 1
        self.agent_pos = new_pos
        self.state = self._get_state(new_pos)

        # Define reward and done conditions
        if new_pos == [self.grid_size - 1, self.grid_size - 1] or new_pos == [self.grid_size - 1, 0] :
            reward = 9999- self.counter
            self.done = True
        elif self._is_hole(new_pos):
            reward = -9999
            self.done = True
        else:
            reward = 0

        return self.state, reward, self.done

    def _get_state(self, pos):
        return pos[0] * self.grid_size + pos[1]

    def _is_hole(self, pos):
        # Define some holes for simplicity (you can customize this)
        holes = self.holes
        return pos in holes

    def is_valid_action(self, pos, action):
        if action == 0 and pos[1] == 0:  # left
            return False
        if action == 1 and pos[0] == self.grid_size - 1:  # down
            return False
        if action == 2 and pos[1] == self.grid_size - 1:  # right
            return False
        if action == 3 and pos[0] == 0:  # up
            return False
        return True

    def render(self):
        grid = np.zeros((self.grid_size, self.grid_size), dtype=str)
        grid[:] = '.'
        grid[self.grid_size - 1, self.grid_size - 1] = 'G'  # Goal
        grid[self.grid_size - 1, 0] = 'G'  # Goal
        for hole in self.holes:
            grid[hole[0], hole[1]] = 'H'  # Hole
        grid[self.agent_pos[0], self.agent_pos[1]] = 'A'  # Agent
        print("\n".join(" ".join(row) for row in grid))
        print()

# Huấn luyện agent và lưu ma trận Q
class FrozenLakeAgent:
    def __init__(self, env, alpha=0.8, gamma=0.95, epsilon=0.1, num_episodes=5000):
        self.env = env
        self.alpha = alpha  # Tốc độ học tập
        self.gamma = gamma  # Hệ số giảm dần
        self.epsilon = epsilon  # Xác suất chọn hành động ngẫu nhiên
        self.num_episodes = num_episodes  # Số tập huấn luyện
        self.Q = np.zeros([env.n_states, env.n_actions])

    def choose_action(self, state):
        valid_actions = [action for action in range(self.env.n_actions) if self.env.is_valid_action(self._pos_from_state(state), action)]
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)  # Chọn hành động ngẫu nhiên trong các hành động hợp lệ
        else:
            q_values = self.Q[state, :]
            q_values[[action for action in range(self.env.n_actions) if action not in valid_actions]] = -float('inf')  # Thiết lập giá trị Q của các hành động không hợp lệ thành -inf
            max_q = np.max(q_values)
            max_actions = [action for action in range(self.env.n_actions) if q_values[action] == max_q]
            return random.choice(max_actions)  # Chọn ngẫu nhiên một hành động trong các hành động có giá trị Q lớn nhất

    def train(self):
        for episode in range(self.num_episodes):
            state = self.env.reset()
            done = False

            while not done:
                action = self.choose_action(state)
                next_state, reward, done = self.env.step(action)

                # Cập nhật giá trị Q
                old_value = self.Q[state, action]
                next_max = np.max(self.Q[next_state, :])
                self.Q[state, action] = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)

                # Chuyển sang trạng thái tiếp theo
                state = next_state

        # Lưu ma trận Q vào file
        with open("q_table.pkl", "wb") as f:
            pickle.dump(self.Q, f)

        print("Ma trận Q đã được lưu vào file q_table.pkl")
        print(self.Q)

    def load_q_table(self, filepath="q_table.pkl"):
        # Tải ma trận Q từ file
        with open(filepath, "rb") as f:
            self.Q = pickle.load(f)
        print("Ma trận Q đã được tải từ file q_table.pkl")
        print(self.Q)

    def run(self):
        state = self.env.reset()
        self.env.render()
        done = False

        while not done:
            action = np.argmax(self.Q[state, :])
            next_state, reward, done = self.env.step(action)
            time.sleep(1)
            os.system("clear")
            self.env.render()
            state = next_state

            if done:
                if reward > 0:
                    print("Agent đạt được phần thưởng!")
                else:
                    print("Agent rơi vào hố băng!")
                break

    def _pos_from_state(self, state):
        return [state // self.env.grid_size, state % self.env.grid_size]

# Sử dụng class
env = FrozenLake()
agent = FrozenLakeAgent(env,alpha=0.15)
agent.train()
agent.load_q_table()
os.system("clear")
agent.run()
print(agent.Q)