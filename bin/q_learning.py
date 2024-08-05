import numpy as np

# Khởi tạo môi trường
grid_size = 3
start_state = (0, 0)
goal_state = (2, 2)
actions = ['up', 'down', 'left', 'right']
num_actions = len(actions)
num_states = grid_size * grid_size

# Hàm chuyển trạng thái
def next_state(state, action):
    x, y = state
    if action == 'up' and x > 0:
        x -= 1
    elif action == 'down' and x < grid_size - 1:
        x += 1
    elif action == 'left' and y > 0:
        y -= 1
    elif action == 'right' and y < grid_size - 1:
        y += 1
    return (x, y)

# Hàm nhận phần thưởng
def get_reward(state):
    if state == goal_state:
        return 10
    else:
        return -1

# Khởi tạo bảng giá trị Q
Q_table = np.zeros((num_states, num_actions))

# Tham số Q-learning
alpha = 0.1  # learning rate
gamma = 0.9  # discount factor
epsilon = 0.1  # exploration rate
num_episodes = 10000

# Hàm chuyển đổi trạng thái thành chỉ số
def state_to_index(state):
    return state[0] * grid_size + state[1]

# Thuật toán Q-learning
for episode in range(num_episodes):
    state = start_state
    while state != goal_state:
        state_idx = state_to_index(state)
        
        # Chọn hành động (epsilon-greedy)
        if np.random.uniform(0, 1) < epsilon:
            action_idx = np.random.choice(num_actions)
        else:
            action_idx = np.argmax(Q_table[state_idx])

        action = actions[action_idx]
        next_state_ = next_state(state, action)
        reward = get_reward(next_state_)
        next_state_idx = state_to_index(next_state_)

        # Cập nhật giá trị Q
        Q_table[state_idx, action_idx] = Q_table[state_idx, action_idx] + alpha * (reward + gamma * np.max(Q_table[next_state_idx]) - Q_table[state_idx, action_idx])

        state = next_state_

# Chính sách tối ưu
optimal_policy = np.argmax(Q_table, axis=1).reshape(grid_size, grid_size)
action_map = np.array(actions)
optimal_policy = action_map[optimal_policy]

print("Bảng giá trị Q:")
print(Q_table)
print("\nChính sách tối ưu:")
print(optimal_policy)
