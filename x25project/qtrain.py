from qclass import Env, X25Agent


env = Env()
x25 = X25Agent(env)
x25.train(int(input("number: ")))