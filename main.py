import gym
from trainer import Trainer

env = gym.make("CartPole-v0")
trainer = Trainer(env)
trainer.train()
