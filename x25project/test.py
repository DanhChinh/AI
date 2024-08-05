from handleFrame import readUniqueData, convertToL2Numpy
import os



def make_actions():
    actions = []
    ratios = [4, 80]
    numbers = [i for i in range(100)]
    for ratio in ratios:
        for number in numbers:
            actions.append(f"{ratio}_{number}")
    return actions

def make_states(numpyArray):
    states = []
    for i in numpyArray:
        print(i)
    pass
class X25Agent:
    def __init__(self, env, alpha=0.8, gamma=0.95, epsilon=0.1, num_episodes=10000):
            self.env = env
            self.alpha = alpha  # Tốc độ học tập
            self.gamma = gamma  # Hệ số giảm dần
            self.epsilon = epsilon  # Xác suất chọn hành động ngẫu nhiên
            self.num_episodes = num_episodes  # Số tập huấn luyện
            if os.path.exists('q_table.csv'):
                self.Q = pd.read_csv('q_table.csv',index_col=0)
            else:
                self.Q = pd.DataFrame(0, index=self.states, columns=self.actions)


data = readUniqueData()
data = convertToL2Numpy(data)
print(data)
# make_states(data)

def arrToState(arr=[]):
    import numpy
    arr = ['86', '49', '13', '24', '07', '36', '24', '78', '14', '86', '08', '37']
    arr = numpy.array(arr)
    db = arr[0]
    arr = numpy.sort(arr.astype(int))
    arr = numpy.unique(arr)
    print(arr)
    # print(arr.count('86'))

# arrToState()


#fix convertToL2Numpy // %

# print(34713%100)