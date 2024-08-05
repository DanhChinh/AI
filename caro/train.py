import numpy as np
import random
import pickle

class TicTacToe:
    def __init__(self, size=3, isPlayerMove=False):
        self.board = np.zeros((size, size), dtype=int)
        self.done = False
        self.size = size

    def reset(self):
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.done = False
        self.isPlayerMove = False
        return self.board

    def is_valid_action(self, action):
        return self.board[action] == 0

    def step(self, action, player):
        if self.done:
            raise Exception("Game is already finished. Please reset the environment.")

        if not self.is_valid_action(action):
            raise Exception("Invalid action")

        self.board[action] = player
        reward, self.done = self.check_game_status(player)
        return self.board, reward, self.done

    def check_game_status(self, player):
        for i in range(3):
            if np.all(self.board[i, :] == player) or np.all(self.board[:, i] == player):
                return 1, True
        if np.all(np.diag(self.board) == player) or np.all(np.diag(np.fliplr(self.board)) == player):
            return 1, True
        if np.all(self.board != 0):
            return 0, True
        return 0, False

class QLearningAgent:
    def __init__(self, alpha=0.8, gamma=0.95, epsilon=0.1, num_episodes=5000):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_episodes = num_episodes
        self.q_table = {}

    def choose_action(self, state, valid_actions):
        if np.random.uniform(0, 1) < self.epsilon:
            return random.choice(valid_actions)
        else:
            state_str = str(state)
            if state_str not in self.q_table:
                self.q_table[state_str] = np.zeros(9)
            q_values = self.q_table[state_str]
            max_q = np.max(q_values[valid_actions])
            max_actions = [a for a in valid_actions if q_values[a] == max_q]
            return random.choice(max_actions)

    def update_q_value(self, state, action, reward, next_state, next_valid_actions):
        state_str = str(state)
        next_state_str = str(next_state)

        if state_str not in self.q_table:
            self.q_table[state_str] = np.zeros(9)
        if next_state_str not in self.q_table:
            self.q_table[next_state_str] = np.zeros(9)

        old_value = self.q_table[state_str][action]
        next_max = np.max(self.q_table[next_state_str][next_valid_actions])
        self.q_table[state_str][action] = old_value + self.alpha * (reward + self.gamma * next_max - old_value)

    def train(self, env):
        for episode in range(self.num_episodes):
            state = env.reset()
            done = False
            player = 1

            while not done:
                valid_actions = [i for i in range(9) if env.is_valid_action((i//3, i%3))]
                action = self.choose_action(state, valid_actions)
                next_state, reward, done = env.step((action//3, action%3), player)

                if not done:
                    next_valid_actions = [i for i in range(9) if env.is_valid_action((i//3, i%3))]
                    self.update_q_value(state, action, reward, next_state, next_valid_actions)
                    state = next_state
                    player = -player
                else:
                    self.update_q_value(state, action, reward, next_state, [])

    def save_q_table(self, filename):
        with open(filename, "wb") as f:
            pickle.dump(self.q_table, f)

    def load_q_table(self, filename):
        with open(filename, "rb") as f:
            self.q_table = pickle.load(f)

# Khởi tạo môi trường và agent
env = TicTacToe()
print(env.board)
# agent = QLearningAgent()

# Huấn luyện agent
# agent.train(env)

# Lưu Q-table
# agent.save_q_table("tic_tac_toe_q_table.pkl")
