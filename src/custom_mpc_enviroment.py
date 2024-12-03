import gym
import numpy as np
from gym import spaces
from pkg_ddpg_td3.environment.components.component import Component
from typing import Union, List, Tuple

from pkg_ddpg_td3.environment.environment import TrajectoryPlannerEnvironment

class MPCTrainingEnviroment(TrajectoryPlannerEnvironment):
    """
    Enviroment for training with mpc
    """


    def __init__(self, generate_map):
        super().__init__()