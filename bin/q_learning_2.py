import numpy as np
import gym

# Tạo môi trường
env = gym.make("FrozenLake-v0")

# Thiết lập tham số
alpha = 0.8  # Tốc độ học tập
gamma = 0.95  # Hệ số giảm dần
epsilon = 0.1  # Xác suất chọn hành động ngẫu nhiên (thay vì hành động tốt nhất hiện tại)
num_episodes = 2000  # Số tập huấn luyện

# Khởi tạo ma trận Q với kích thước (số trạng thái, số hành động)
Q = np.zeros([env.observation_space.n, env.action_space.n])

# Hàm chọn hành động dựa trên chính sách epsilon-greedy
def choose_action(state):
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Chọn hành động ngẫu nhiên
    else:
        return np.argmax(Q[state, :])  # Chọn hành động có giá trị Q lớn nhất

# Q-learning
for episode in range(num_episodes):
    state = env.reset()  # Đặt lại môi trường và lấy trạng thái ban đầu
    done = False

    while not done:
        action = choose_action(state)
        next_state, reward, done, _ = env.step(action)

        # Cập nhật giá trị Q
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])

        # Chuyển sang trạng thái tiếp theo
        state = next_state

# Hiển thị ma trận Q đã học
print("Ma trận Q đã học:")
print(Q)

# Kiểm tra chính sách đã học
state = env.reset()
env.render()
done = False
while not done:
    action = np.argmax(Q[state, :])
    state, reward, done, _ = env.step(action)
    env.render()
