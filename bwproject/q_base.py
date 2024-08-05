import numpy as np
import pandas as pd
import random
import itertools
import sys 
import os
def stop():
    sys.exit()
def make_indexs(a, b, length=1):
    combinations = [''.join(x) for x in itertools.product(a, repeat=length)]
    return [comb + "_" + last_char for comb in combinations for last_char in b]

class BlackWhite:
    def __init__(self):
        self.reset()

    def reset(self):
        self.points = 100
        self.prediction = None
        self.spent_points = 0
        self.done = False

    def step(self, action, result):
        if self.done:
            raise Exception("Game is already finished. Please reset the environment.")

        [prediction,percentage] = action.split('_')
        reward = int(self.points*float(percentage))
        if prediction == "left":
            prediction = '1'
        else:
            prediction = '0'
        if prediction != result:
            # print(prediction,result)
            reward = -reward
        self.points += reward
        if self.points<10 or self.points>200:
            self.done = True
        return reward, self.done
  



# Huấn luyện agent và lưu ma trận Q
class BlackWhiteAgent:
    def __init__(self, env, alpha=0.8, gamma=0.95, epsilon=0.1, num_episodes=10000):
        self.env = env
        self.alpha = alpha  # Tốc độ học tập
        self.gamma = gamma  # Hệ số giảm dần
        self.epsilon = epsilon  # Xác suất chọn hành động ngẫu nhiên
        self.num_episodes = num_episodes  # Số tập huấn luyện
        self.lengthhs = 5
        self.states = make_indexs(['0','1'],['W','L'],self.lengthhs)
        self.actions = make_indexs(['left', 'right'],['10','20','40'])
        self.history = random.choices(['1', '0'], k=self.lengthhs+2)
        self.result = self.history[-1]#fix
        print("history", self.history)
        if os.path.exists('data.csv'):
            self.Q = pd.read_csv('data.csv',index_col=0)
        else:
            self.Q = pd.DataFrame(0, index=self.states, columns=self.actions)
    def get_state(self,result):
        state = ''.join(self.history[-self.lengthhs:])
        if self.history[-1] == result:
            return state+"_W"
        return state+"_L"

    def choose_action(self,state):
        if random.uniform(0, 1) < self.epsilon:
            return random.choice(self.actions)  
        else:
            return self.Q.loc[state].idxmax()

    def train(self):
        for episode in range(self.num_episodes):
            self.env.reset()
            player = self.env
            result = random.choice(['0','1'])
            state = self.get_state(result) 
            while not player.done:
                action = self.choose_action(state)
                result = random.choice(['1','0'])
                self.history.append(result)
                reward, done = self.env.step(action,result)
                next_state = self.get_state(result)
                # Cập nhật giá trị Q
                old_value = self.Q.loc[state, action]
                next_max = self.Q.loc[state].max()
                self.Q.loc[state, action] = int((1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max))

                # Chuyển sang trạng thái tiếp theo
                state = next_state

        # Lưu ma trận Q vào file
        self.Q.to_csv('data.csv',index=True)

    def load_q_table(self, filepath="q_table.pkl"):
        # Tải ma trận Q từ file
        return pd.read_csv('data.csv',index_col=0)

    


# Sử dụng class
env = BlackWhite()
agent = BlackWhiteAgent(env,alpha=0.15)
agent.train()
Q =  agent.load_q_table()
print(Q)
print()

def get_state(history, iseque):

    return ''.join(history[-5:])+"_W" if iseque else ''.join(history[-5:])+"_L"
points = 100
result = random.choice(['0','1'])
prediction = result
history = random.choices(['1', '0'], k=5)
while points>0 and points<200:
    state = get_state(history, prediction==result) 
    action = Q.loc[state].idxmax()
    print(state,action)
    [prediction,percentage] = action.split('_')
    if prediction == "left":
        prediction = '1'
    else:
        prediction = '0'
    reward = int(points*float(percentage)/100)
    if reward ==0:
        break
    result = random.choice(['0','1'])
    history.append(result)
    if prediction == result:
        points+=reward
    else:
        points-=reward
    print(f"{reward}$->{prediction}: {result}, {points}")