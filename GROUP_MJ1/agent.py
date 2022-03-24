import numpy as np

class Agent():
  '''The agent class that is to be filled.
     You are allowed to add any method you
     want to this class.
  '''

  def __init__(self, env_specs):
    self.env_specs = env_specs

  def load_weights(self, root_path):
    # Add root_path in front of the path of the saved network parameters
    # For example if you have weights.pth in the GROUP_MJ1, do `root_path+"weights.pth"` while loading the parameters
    pass

  def act(self, curr_obs, mode='eval'):
    return self.env_specs['action_space'].sample()

  def update(self, curr_obs, action, reward, next_obs, done, timestep):
    pass
