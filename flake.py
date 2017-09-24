import pdb
import random
import sys,tty,termios

import torch
import torch.nn as nn
from torch.autograd import Variable

import gym
import numpy as np
import matplotlib.pyplot as plt
from gym.envs.registration import register

class _Getch:
    def __call__(self):
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            ch = sys.stdin.read(3)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return ch

inkey = _Getch()

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

arrow_keys = {
        '\x1b[A': UP,
        '\x1b[B': DOWN,
        '\x1b[C': RIGHT,
        '\x1b[D': LEFT}

def rargmax(vector):
    m = np.amax(vector)
    indices = np.nonzero(vector == m)[0]
    return random.choice(indices)

#register(
#        id='FrozenLake-v3',
#        entry_point='gym.envs.toy_text:FrozenLakeEnv',
#        kwargs={'map_name': '4x4', 'is_slippery': True})

#state = env.reset()
#while True:
#    key = inkey()
#    if not key in arrow_keys.keys():
#        print('Abort for not arrow key')
#        break
#    action = arrow_keys[key]
#    state, reward, done, info = env.step(action)
#    print('state: {}, reward: {}'.format(state, reward))
#    env.render()
#    if done:
#        print('Finished with reward: {}'.format(reward))
#        break
#exit()

def one_hot(x, input_size):
    return np.identity(input_size)[x: x+1]

class Net(nn.Module):
    def __init__(self, input_size, output_size, hidden_dims):
        super(Net, self).__init__()
        self.layers = []
        
        prev_dim = input_size
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.layers.append(nn.ReLU(True))
            prev_dim = hidden_dim
        self.layers.append(nn.Linear(prev_dim, output_size))

        self.layer_module = nn.ModuleList(self.layers)
    
    def forward(self, x):
        out = x
        for layer in self.layer_module:
            out = layer(out)
        return out


random.seed(1234)
env = gym.make("FrozenLake-v0")

input_size = env.observation_space.n
output_size = env.action_space.n
net = Net(input_size, output_size, [32, 64, 128, 256])

m = nn.Linear(input_size, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(m.parameters(), lr=0.1)

#Q = np.zeros([env.observation_space.n, env.action_space.n])
num_episodes = 2000
alpha = 0.5
gamma = 0.99


rList = []
for i in range(num_episodes):

    state = env.reset()
    e = 1. / ((i // 50) + 10)
    rAll = 0
    done = False
    local_loss = []

    while not done:

        Qs = m(Variable(torch.Tensor(one_hot(state, input_size))))

        if np.random.rand(1) < e:
            action = env.action_space.sample() 
        else:
            action = torch.topk(Qs, 1)[1].data[0][0]

#        action = np.argmax(Q[state, :] +
#                           np.random.randn(1, env.action_space.n) / (i + 1))

        state_, reward, done, _ = env.step(action)

        if done:
            Qs[0, action] = reward
        else:
            Qs1 = m(Variable(torch.Tensor(one_hot(state_, input_size))))
            Qs[0, action] = reward + gamma * Qs1.max()


        input_var = Variable(torch.Tensor(one_hot(state, input_size)))
        Qpred = m(input_var)

        loss = criterion(Qpred, Qs.detach())

        m.zero_grad()
        loss.backward()
        
        optimizer.step()


#        Q[state, action] = (1-alpha) * Q[state, action] + \
#                alpha * (reward + gamma * np.max(Q[state_, :]))
#        Q[state, action] +=\
#            alpha * (reward + gamma * np.max(Q[state_, :]) - Q[state, action])

        rAll += reward
        state = state_
    
    rList.append(rAll)

print("Success rate: {}".format(str(sum(rList)/num_episodes)))
print("Final Q-Table Values")
print("Left Down Right Up")
print(Qs)
plt.bar(range(len(rList)), rList, color='blue')
plt.show()


