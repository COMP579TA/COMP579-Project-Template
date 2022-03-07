# COMP579 Project Template

The environment dependencies can be found in `environment.yml`. Make sure to also install mujoco 210 manually.

After creating your conda environment with `environment.yml`, activate the environment and do `conda env config vars set LD_LIBRARY_PATH=/usr/local/pkgs/cuda/latest/lib64:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia-460:/usr/lib/nvidia`. Note that `nvidia-460` depends on your machine.

The agent can be trained by running `python3 train_agent.py --group GROUP_MJ1`
