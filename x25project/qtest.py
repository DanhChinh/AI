from qclass import Env, X25Agent, arrToState
from handleFrame import readDataL2




data = readDataL2("dataTest.csv")
env = Env()
x25 = X25Agent(env)
done = False
index = -1
while not done:
    if index>=len(data)-2:
        break
    else:
        index+=1
    state = arrToState(data[index])
    action = x25.choose_action(state)
    reward, _ = env.step(action, data[index+1])
    print(reward, env.point)
