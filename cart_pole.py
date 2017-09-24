import pdb
import random
from collections import deque

import torch
import torch.nn as nn
from torch.autograd import Variable

import gym
import matplotlib.pyplot as plt
import numpy as np

import model


def preprocess_input(x, volatile):
    return Variable(torch.FloatTensor(x), volatile=volatile)

def simple_replay_train(DQN, train_batch):
#    x_stack = np.empty(0).reshape(0, DQN.input_size)
#    y_stack = np.empty(0).reshape(0, DQN.output_size)

    x_stack = []
    y_stack = []
    for state, action, reward, next_state, done in train_batch:
        Q = DQN(preprocess_input(state, volatile=True))

        if done:
            Q[0, action] = reward
        else:
            Q[0, action] = reward + gamma * torch.max(DQN(preprocess_input(next_state, volatile=True)))

        x_stack.append(state)
        y_stack.append(Q.data)

    x_stack = np.stack(x_stack)
    y_stack = np.stack(y_stack)

    output = DQN(x_stack)
    loss = criterion(output, y_stack) 

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
        
    return loss
    


env = gym.make('CartPole-v0')

input_size = env.observation_space.shape[0]
output_size = env.action_space.n
model = model.DQN(input_size, output_size, [10])

criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Make replay buffer
REPLAY_MEMORY_SIZE = 50000 
replay_buffer = deque()



gamma = 0.9
num_episodes = 5000
rList = []
loss_list = []
model.train()

for i in range(num_episodes):
    
    e = 1. / ((i // 10) + 1) 
    step_count = 0
    state = env.reset()
    done = False

    while not done:
        step_count += 1
        Qs = model(Variable(torch.FloatTensor(state.reshape(1, -1))))
        if random.random() < e:
            action = env.action_space.sample()
        else:
            action = Qs.topk(1)[1].data[0][0]

        state_1, reward, done, _ = env.step(action)
        replay_buffer.append((state, action, reward, state_1, done))
        if len(replay_buffer) > REPLAY_MEMORY_SIZE:
            replay_buffer.popleft()

        if episode % 10 == 1:   # train every 10 episodes.
            # Get a random batch of experiences.
            for _ in range(50):
                # Minibatch works better
                minibatch = random.sample(replay_buffer, 10)
                loss, _ = simple_replay_train(mainDQN, minibatch)

        Q_target = Variable(Qs.data.clone(), volatile=True)
        if done:
            Q_target[0, action] = -100
        else:
            Qs_1 = model(Variable(torch.FloatTensor(state_1.reshape(1, -1)),
                              volatile=True))
            pdb.set_trace()
            Q_target[0, action] = reward + gamma * torch.max(Qs_1)

        Q_target.volatile = False
        loss = criterion(Qs, Q_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        state = state_1

    rList.append(step_count)
    loss_list.append(loss.data[0])
    if (i+1) % 100 == 0:
        print('[{}/{}]\tLoss: {}'.format(i, num_episodes, loss.data[0]))
    if len(rList) > 10 and np.mean(rList[-10:]) > 500:
        break

print("Score over time: {}".format(sum(rList) / num_episodes))
fig = plt.figure()
ax1 = plt.subplot(121)
ax2 = plt.subplot(122)
ax1.bar(range(len(rList)), rList, color="blue")
ax2.bar(range(len(loss_list)), loss_list, color="red")
plt.show()

exit()

observation = env.reset()
reward_sum = 0
while True:
    env.render()

    x = torch.FloatTensor(np.reshape(observation, [q, input_size]))
    Qs = m(Varaible(x))
    a = Qs.topk(1)[1].data[0][0]

    observation, reward, done, _ = env.step(a)
    reward_sum += reward
    if done:
        print("Total score: {}".format(reward_sum))
        break




