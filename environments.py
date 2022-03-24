import gym
import numpy as np


class JellyBeanEnv(gym.Wrapper):
  '''The JellyBean Environment Wrapper.'''

  def __init__(self, env):
    super().__init__(env)
    self.env = env

  def reset(self):
    self.env.reset()

  def step(self, action):
    next_obs, reward, done, info = self.env.step(action)
    return next_obs, reward, done, info

  def seed(self, seed):
    self.env.seed(seed)
    self.env.action_space.seed(seed)
    self.env.scent_space.seed(seed)
    self.env.vision_space.seed(seed)
    self.env.feature_space.seed(seed)

class MujocoEnv(gym.Wrapper):
  '''The Mujoco Environment Wrapper.'''

  def __init__(self, env):
    super().__init__(env)
    self.env = env

  def reset(self):
    return self.env.reset()

  def step(self, action):
    next_obs, reward, done, info = self.env.step(action)
    return next_obs, reward, done, info

  def seed(self, seed):
    self.env.seed(seed)
    self.env.action_space.seed(seed)
    self.env.observation_space.seed(seed)
