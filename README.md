# COMP579 Project Template
## Virtual Environment
Create a virtual environment with the dependencies found in `environment.yml`. 

```
git clone https://github.com/COMP579TA/COMP579-Project-Template
cd COMP579-Project-Template
conda env create environment.yml -n my-venv
conda activate my-venv
```

## Mujoco Installation
We'll be using mujoco210 in this project.
https://github.com/deepmind/mujoco/releases/tag/2.1.0
This page contains the mujoco210 releases.
Download the distribution compatible to your OS and unzip it inside ~/.mujoco/

After activating the virtual environment, change the environment variable as:
```
conda env config vars set LD_LIBRARY_PATH=/usr/local/pkgs/cuda/latest/lib64:$HOME/.mujoco/mujoco210/bin:/usr/lib/nvidia-460:/usr/lib/nvidia`
```

Note that the GPU driver `nvidia-460` is only applicable for machines with GPUs and is machine specific.
For installing mujoco on a CPU only machine do this:

```
fill this part
```

_A section for google colab?_

## Set the origin to your group level git repository
After creating you own private git repository with this template, add this github account as a collaborator. We will be suing this account to pull the submission folder from your repository every time we refresh the leaderboard.

## Training the agent
The agent can be trained by running `python3 train_agent.py --group GROUP_MJ1`
