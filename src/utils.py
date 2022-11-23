from unityagents import UnityEnvironment
import numpy as np
import torch


def print_device_info():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using Pytorch version {torch.__version__} on device {device}")
    #Additional Info when using cuda
    if device.type == 'cuda':
        print(torch.cuda.get_device_name(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

def print_env_info(env):
    # get the default brain
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    print(f"brain_name: {brain_name}")
    print(f"brain: {brain}")

    # reset the environment
    unity_env = env.reset(train_mode=True)[brain_name]

    action_size = brain.vector_action_space_size
    state = unity_env.vector_observations[0]
    state_size = len(state)

    print(f"action_size: {action_size}")
    print(f"state_size: {state_size}")
    print(f"First observation/state: {state}")

def get_env(visual_mode=False):
    if visual_mode:
        return UnityEnvironment(file_name="src/Tennis_Linux/Tennis.x86_64")
    else:
        return UnityEnvironment(file_name="src/Tennis_Linux_NoVis/Tennis.x86_64")