from theano.sandbox import cuda
cuda.use('gpu0')
import cv2
import numpy as np
import gym
from gym.spaces.box import Box
import universe
from universe import vectorized
from universe.wrappers import BlockingReset, GymCoreAction, EpisodeID, Unvectorize, Vectorize, Vision, Logger
from universe import spaces as vnc_spaces
from universe.spaces.vnc_event import keycode
from dqn import Agent
import time
from util import *

env = gym.make('gym-core.PongDeterministic-v3')
env = Vision(env)
env = Logger(env)
env = BlockingReset(env)
env = GymCoreAction(env)
env = AtariRescale42x42(env)
env = EpisodeID(env)
env = DiagnosticsInfo(env)
env = Unvectorize(env)
fps = env.metadata['video.frames_per_second']
env.configure(remotes=1, start_timeout=15*60, fps=fps)

shape = env.observation_space.shape
action_space = env.action_space.n
agent = Agent(state_size=shape, number_of_actions=action_space, save_name='mytest')

for e in range(10000):
    observation_n = env.reset()
    agent.new_episode()
    done_n = False
    while not done_n:
        action_n, values = agent.act(observation_n)
        observation_n, reward_n, done_n, info = env.step(action_n)
        env.render()
