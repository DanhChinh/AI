from handleFrame import readDataL2
import os
import numpy as np
import pandas as pd
from scipy import stats



def make_actions():
    actions = []
    ratios = [4, 80]
    numbers = [i for i in range(100)]
    for ratio in ratios:
        for number in numbers:
            actions.append(f"{ratio}_{number}")
    return actions

def arrToState(arr):
    #mode, phuongsai, dolechchuan,...
    db = arr[0]
    mean = np.mean(arr)
    median = np.median(arr)
    mode = stats.mode(arr)
    return f"{int(mean)}_{int(median)}_{mode[1]}"

def make_states(numpyArray):
    states = []
    for arr in numpyArray:
        state = arrToState(arr)
        if state not in states:  # Kiểm tra trùng lặp
            states.append(state)
    return states
class X25Agent:
    def __init__(self, env, alpha=0.1, gamma=0.95, epsilon=0.15, num_episodes=10000):
            self.env = env
            self.alpha = alpha  # Tốc độ học tập
            self.gamma = gamma  # Hệ số giảm dần
            self.epsilon = epsilon  # Xác suất chọn hành động ngẫu nhiên
            self.num_episodes = num_episodes  # Số tập huấn luyện
            self.numpyData = readDataL2()
            self.actions = make_actions()
            self.states = make_states(self.numpyData)
            if os.path.exists('q_table.csv'):
                self.Q = pd.read_csv('q_table.csv',index_col=0)
            else:
                self.Q = pd.DataFrame(0, index=self.states, columns=self.actions)

x25 = X25Agent(1)
