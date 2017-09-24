import pdb
import random
from collections import deque

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

import models


class Trainer():

    def __init__(self, env):
        
        self.env = env
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.n
        self.gamma = 0.9
        self.num_episodes = 5000
        self.replay_buffer = deque()
        self.MEMORY_SIZE = 50000
    
        self.model = models.DQN(self.input_size, self.output_size, [10])
        self.target_Q = models.DQN(self.input_size, self.output_size, [10])
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01)

    def train(self):

        self.model.train()
        for ep in range(self.num_episodes):

            step_count = 0
            state = self.env.reset()
            done = False

            while not done:
                a = self.choose_action(ep, state)

                next_state, reward, done, _ = self.env.step(a)
                self.replay_buffer.append((state, a, reward, next_state, done))
                if len(self.replay_buffer) > self.MEMORY_SIZE:
                    self.replay_buffer.popleft()

                step_count += 1
                if step_count > 10000:
                    break

                state = next_state

            print("Episodes: [{}/{}], steps: {}".format(
                            ep, self.num_episodes, step_count)) 

            if (ep + 1) % (10 + 1) == 0: # train every 10 episodes.
                # Get a random batch of experiences.
                for _ in range(50):
                    # batch works better
                    batch = random.sample(self.replay_buffer, 10)
                    loss = self.simple_replay_train(batch)

                self.target_Q.load_state_dict(self.model.state_dict())

        self.bot_play(self.model)


    def choose_action(self, ep, state):
        epsilon = 1. / ((ep // 10) + 1)
        if np.random.rand(1) < epsilon:
            action = self.env.action_space.sample()
        else:
            Qs = self.target_Q(self._preprocess(state, volatile=True))
            action = Qs.topk(1)[1].data[0][0]

        return action


    def _preprocess(self, x, volatile):
        return Variable(torch.FloatTensor(x.reshape(1, -1)), volatile=volatile)


    def simple_replay_train(self, train_batch):

        x_stack = []
        y_stack = []
        for state, action, reward, next_state, done in train_batch:
            Q = self.target_Q(self._preprocess(state, volatile=True))

            if done:
                Q[0, action] = -100 
            else:
                Q_1 = self.target_Q(self._preprocess(next_state, volatile=True))
                Q[0, action] = reward + self.gamma * Q_1.max() 

            x_stack.append(state)
            y_stack.append(Q.data.numpy().reshape(-1))

        x_stack = Variable(torch.FloatTensor(np.stack(x_stack)))
        y_stack = Variable(torch.FloatTensor(np.stack(y_stack)))

        output = self.model(x_stack)
        loss = self.criterion(output, y_stack)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
            
        return loss

    
    def bot_play(self, mainDQN):
        s = self.env.reset()
        reward_sum = 0
        while True:
            self.env.render()
            Q = DQN(self._preprocess(state, volatile=True))
            a = Q.topk(1)[1].data[0][0]
            s, reward, cone, _ = self.env.step(a)
            reward_sum += reward
            if done or reward_sum > 10000:
                print("Total score : {}".format(reawrd_sum))
                break
