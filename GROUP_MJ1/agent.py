import numpy as np

class Agent():
  '''The agent class that is to be filled.
     You are allowed to add any method you
     want to this class.
  '''

  def __init__(self, env):
    self.env = env

  def load_weights(self):
    pass

  def act(self, curr_obs, mode='eval'):
    return self.env.action_space.sample()

  def update(self, curr_obs, action, reward, next_obs, done, timestep):
    pass
