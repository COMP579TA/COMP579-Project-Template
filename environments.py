import gym
import numpy as np


class JellyBeanEnv(gym.Wrapper):
  '''The JellyBean Environment Wrapper.
     Not to be edited!
  '''

  def __init__(self, env):
    super().__init__(env)
    self.env = env

  def reset(self):
    self.env.reset()

  def step(self, action):
    next_obs, reward, done, info = self.env.step(action)
    return next_obs, reward, done, info


class MujocoEnv(gym.Wrapper):
  '''The Mujoco Environment Wrapper.
     Not to be edited!
  '''

  def __init__(self, env):
    super().__init__(env)
    self.env = env

  def reset(self):
    return self.env.reset()

  def step(self, action):
    next_obs, reward, done, info = self.env.step(action)
    return next_obs, reward, done, info
