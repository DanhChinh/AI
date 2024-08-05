import warnings
import numpy as np
import gym
import pickle

# Tắt cảnh báo DeprecationWarning cho np.bool8
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*np.bool8.*")

# Tạo môi trường với chế độ render
env = gym.make("FrozenLake-v1", render_mode=None)  # Không cần render trong quá trình huấn luyện

# Thiết lập tham số
alpha = 0.15  # Tốc độ học tập
gamma = 0.9  # Hệ số giảm dần
epsilon = 0.2  # Xác suất chọn hành động ngẫu nhiên (thay vì hành động tốt nhất hiện tại)
num_episodes = 50000  # Số tập huấn luyện

# Khởi tạo ma trận Q với kích thước (số trạng thái, số hành động)
Q = np.zeros([env.observation_space.n, env.action_space.n])
# with open("q_table.pkl", "rb") as f:
#     Q = pickle.load(f)
# print("read q_table.pkl")
print(Q)

# Hàm chọn hành động dựa trên chính sách epsilon-greedy
def choose_action(state):
    state_idx = state if isinstance(state, int) else state[0]  # Trích xuất chỉ số trạng thái
    if np.random.uniform(0, 1) < epsilon:
        return env.action_space.sample()  # Chọn hành động ngẫu nhiên
    else:
        return np.argmax(Q[state_idx, :])  # Chọn hành động có giá trị Q lớn nhất

# Q-learning
for episode in range(num_episodes):
    state, info = env.reset()  # Đặt lại môi trường và lấy trạng thái ban đầu
    state = state if isinstance(state, int) else state[0]  # Trích xuất chỉ số trạng thái
    done = False

    while not done:
        action = choose_action(state)
        next_state, reward, done, truncated, info = env.step(action)
        next_state = next_state if isinstance(next_state, int) else next_state[0]  # Trích xuất chỉ số trạng thái

        # Cập nhật giá trị Q
        Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
        # Chuyển sang trạng thái tiếp theo
        state = next_state

# Lưu ma trận Q vào file
with open("q_table.pkl", "wb") as f:
    pickle.dump(Q, f)

print("Ma trận Q đã được lưu vào file q_table.pkl")
print(Q)

# Đóng môi trường
env.close()

