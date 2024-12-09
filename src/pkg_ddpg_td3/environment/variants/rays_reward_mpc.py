import sys
import os
import pathlib
import gym
import copy
import numpy as np

from packaging import version

from ..components import *
from .. import MapGenerator, MobileRobot
from ..environment import TrajectoryPlannerEnvironment


GYM_0_22_X = version.parse(gym.__version__) >= version.parse("0.22.0")
DYN_OBS_SIZE = 0.8 + 0.8


sys.path.append('../')

from src.main_continous_evaluation import load_mpc, circle_to_rect, est_dyn_obs_positions, ref_traj_filter

from helper_main_continous import generate_map, get_geometric_map, HintSwitcher, Metrics
from pkg_ddpg_td3.utils.per_ddpg import PerDDPG
from typing import List, Tuple

from timer import PieceTimer, LoopTimer



class TrajectoryPlannerEnvironmentRaysRewardMPC(TrajectoryPlannerEnvironment):
    """
    Environment with what the associated report describes as ray and sector
    observations and reward R_2
    """
    def __init__(
        self,
        generate_map: MapGenerator,
        time_step: float = 0.2,
        reference_path_sample_offset: float = 0,
        corner_samples: int = 3,
        num_segments: int = 8,
        collision_reward_factor: float = 10,
        cross_track_reward_factor: float = 0.1,
        speed_reward_factor: float = 0.5,
        reference_speed: float = MobileRobot().cfg.SPEED_MAX * 0.8,
        jerk_factor: float = 0.02,
        angular_jerk_factor: float = 0.02,
    ):
        super().__init__(
            [
                SpeedObservation(),
                AngularVelocityObservation(),
                ReferencePathSampleObservation(1, 0, reference_path_sample_offset),
                ReferencePathCornerObservation(corner_samples),
                SectorAndRayObservation(num_segments, use_memory=True),
                CollisionReward(collision_reward_factor),
                CrossTrackReward(cross_track_reward_factor),
                SpeedReward(speed_reward_factor, reference_speed),
                JerkReward(jerk_factor),
                AngularJerkReward(angular_jerk_factor),
            ],
            generate_map,
            time_step
        )


        cfg_fpath = '/home/valsamu/DRL-Traj-Planner/config/mpc_longiter.yaml'
        self.traj_gen = load_mpc(cfg_fpath, verbose=False)
        self.last_action = np.array([0, 0])

        self.reset()

    def step(self, action: int) -> tuple[dict, float, bool, bool, dict]:


        dyn_obstacle_list = [obs.keyframe.position.tolist() for obs in self.obstacles if not obs.is_static]
        dyn_obstacle_list_poly = [circle_to_rect(obs) for obs in dyn_obstacle_list]
        dyn_obstacle_pred_list = []
        if self.last_dyn_obstacle_list is None:
            self.last_dyn_obstacle_list = dyn_obstacle_list
        for j, dyn_obs in enumerate(dyn_obstacle_list):
            dyn_obstacle_pred_list.append(est_dyn_obs_positions(self.last_dyn_obstacle_list[j], dyn_obs))
        self.last_dyn_obstacle_list = dyn_obstacle_list


        #self.traj_gen.set_current_state(self.agent.state)
        rl_ref = []
        robot_sim:MobileRobot = copy.deepcopy(self.agent)
        robot_sim:MobileRobot
        for j in range(20):
            if j == 0:
                robot_sim.step(action, self.traj_gen.config.ts)
            else:
                robot_sim.step_with_ref_speed(self.traj_gen.config.ts, 1.0)
            rl_ref.append(list(robot_sim.position))

        if dyn_obstacle_list:
            self.traj_gen.update_dynamic_constraints(dyn_obstacle_pred_list)

        original_ref_traj, rl_ref_traj, *_ = self.traj_gen.get_local_ref_traj(np.array(rl_ref))
        filtered_ref_traj = ref_traj_filter(original_ref_traj, rl_ref_traj, decay=1)

        if self.switch.switch(self.traj_gen.state[:2], original_ref_traj.tolist(), filtered_ref_traj.tolist(), 
                              self.geo_map.processed_obstacle_list+dyn_obstacle_list_poly):
            chosen_ref_traj = filtered_ref_traj
        else:
            chosen_ref_traj = original_ref_traj

            

        timer_mpc = PieceTimer()

        try:
            mpc_output = self.traj_gen.get_action(chosen_ref_traj) # MPC computes the action
            mpc_action, pred_states, cost = mpc_output
        except Exception as e:
            print(f'MPC fails: {e}')
        
        last_mpc_time = timer_mpc(4, ms=True)
        action = (mpc_action - self.last_action)/self.time_step
        self.last_action = mpc_action

        return super().step(action)
    
    def reset(self, seed=None, options=None) -> Tuple[dict]:
        self.last_dyn_obstacle_list = None
        reset_return =  super().reset(seed, options)
        self.geo_map = get_geometric_map(self.get_map_description(), inflate_margin=0.7)
        self.traj_gen.update_static_constraints(self.geo_map.processed_obstacle_list)

        init_state = np.array([*self.agent.position, self.agent.angle])
        goal_state = np.array([*self.goal.position, 0])
        ref_path = list(self.path.coords)
        self.traj_gen.initialization(init_state, goal_state, ref_path)
        self.switch = HintSwitcher(10, 2, 10)
        return reset_return

