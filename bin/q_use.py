import gym
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*np.bool8.*")
# Tạo lại môi trường
env = gym.make("FrozenLake-v1", render_mode="human")

# Tải ma trận Q từ file
with open("q_table.pkl", "rb") as f:
    Q = pickle.load(f)

print("Ma trận Q đã được tải từ file q_table.pkl")
# print(Q)

# Đặt lại môi trường và kiểm tra chính sách đã học
state, info = env.reset()
env.render()
done = False

while not done:
    action = np.argmax(Q[state, :])
    next_state, reward, done, truncated, info = env.step(action)
    
    env.render()
    state = next_state

    if done:
        if reward == 1:
            print("Agent đạt được phần thưởng!")
        else:
            print("Agent rơi vào hố băng!")
        break

