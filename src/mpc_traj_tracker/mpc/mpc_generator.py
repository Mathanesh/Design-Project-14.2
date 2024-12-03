import copy
from itertools import product
import os

import numpy as np
import casadi.casadi as cs
from opengen import opengen as og
import torch # or "import opengen as og"

from pkg_ddpg_td3.environment.environment import TrajectoryPlannerEnvironment
from pkg_ddpg_td3.utils.per_ddpg import PerDDPG
from util.mpc_config import Configurator

from typing import Union, List, Callable

'''
File info:
    Name    - [mpc_generator]
    Date    - [Jan. 01, 2021] -> [Aug. 20, 2021]
    Exe     - [No]
File description:
    The MPC module defines the MPC problem with the cost and constraints.
Comments:
    Adjust MAX_SOVLER_TIME accordingly.
'''

MAX_SOVLER_TIME = 5_000_000 # micros (default 5 sec)

#%%## Helper functions ###
def dist_to_points_square(point:cs.MX, points:List[Union[cs.MX, cs.DM]]):
    return cs.sum1((point-points)**2) # sum1 is summing each column

def dist_to_lineseg(point:cs.MX, line_segment:List[Union[cs.MX, cs.DM]]):
    # Ref: https://math.stackexchange.com/questions/330269/the-distance-from-a-point-to-a-line-segment
    (p, s1, s2) = (point[:2], line_segment[0], line_segment[1])
    s2s1 = s2-s1 # line segment
    t_hat = cs.dot(p-s1,s2s1)/(s2s1[0]**2+s2s1[1]**2+1e-16)
    t_star = cs.fmin(cs.fmax(t_hat,0.0),1.0) # limit t
    temp_vec = s1 + t_star*s2s1 - p # vector pointing to closest point
    distance = cs.sqrt(temp_vec[0]**2+temp_vec[1]**2)
    return distance

def path_progress_to_lineseg(point:cs.MX, line_segment:List[Union[cs.MX, cs.DM]]):
    (p, s1, s2) = (point[:2], line_segment[0], line_segment[1])
    s2s1 = s2-s1 # line segment
    t_hat = cs.dot(p-s1,s2s1)/(s2s1[0]**2+s2s1[1]**2+1e-16)
    t_star = cs.fmin(cs.fmax(t_hat,0.0),1.0) # limit t
    distance = cs.sqrt(s2s1[0]**2+s2s1[1]**2)*t_star
    return distance

def inside_ellipses(point:cs.MX, ellipse_param:List[Union[cs.MX, cs.DM]]):
    # Center: (cx, cy), semi-axes: (rx, ry), rotation angle to x axis: ang
    # If inside, return positive value, else return negative value
    x, y = point[0], point[1]
    cx, cy, rx, ry, ang = ellipse_param[0], ellipse_param[1], ellipse_param[2], ellipse_param[3], ellipse_param[4]
    is_inside = 1 - ((x-cx)*cs.cos(ang)+(y-cy)*cs.sin(ang))**2 / (rx+1e-6)**2 - ((x-cx)*cs.sin(ang)-(y-cy)*cs.cos(ang))**2 / (ry+1e-6)**2
    return is_inside

def inside_pollygon(point:cs.MX, b:cs.MX, a0:cs.MX, a1:cs.MX):
    # Each half-space/edge is defined by b - [a0,a1]*[x,y]' > 0
    # If prod(|max(0,all)|)>0, then the point is inside; Otherwise not.
    eq_mtx = cs.horzcat(a0, a1, b)
    result = cs.mtimes(eq_mtx, cs.vertcat(-point[0], -point[1], 1))
    is_inside = 1
    for i in range(result.shape[0]):
        is_inside *= cs.fmax(0, result[i]) ** 2
    return is_inside

def outside_polygon(point:cs.MX, b:list, a0:list, a1:list):
    # Each half-space/edge is defined by b - [a0,a1]*[x,y]' > 0
    # If sum(|min(0,all)|)>0, then the point is outside; Otherwise not.
    eq_mtx = cs.DM([a0, a1, b]).T
    result = cs.mtimes(eq_mtx, cs.vertcat(-point[0], -point[1], 1))
    is_outside = 0
    for i in range(result.shape[0]):
        is_outside += cs.fmin(0, result[i]) ** 2
    return is_outside

def angle_between_lines(l1:List[list], l2:List[list], normalized:bool=False):
    # line (np.array): [[x0 x1], [y0 y1]]
    vec1 = l1[:,1] - l1[:,0]
    vec2 = l2[:,1] - l2[:,0]
    cos_angle = cs.dot(vec1, vec2)
    if not normalized:
        cos_angle /= (cs.norm_2(vec1)*cs.norm_2(vec2) + 1e-10)
    else:
        cos_angle = cs.fmin(cos_angle,  1.0-1e-10)
        cos_angle = cs.fmax(cos_angle, -1.0+1e-10)
    angle = cs.acos(cos_angle) * cs.sign(vec2[0]*vec1[1]-vec2[1]*vec1[0]) # sign +-1
    return angle

#%%## Define the meta cost functions here ###
def cost_inside_polygon(point:Union[cs.MX, cs.DM], b:cs.MX, a0:cs.MX, a1:cs.MX, weight:float=1):
    indicator = inside_pollygon(point, b, a0, a1) # indicator<0, if outside pollygon
    cost = indicator * weight
    return cost

def cost_inside_ellipses(point:Union[cs.MX, cs.DM], ellipse_param:List[Union[cs.MX, cs.DM]], weight:float=1):
    if len(ellipse_param) > 5:
        alpha = ellipse_param[5]
    else:
        alpha = 1
    indicator = inside_ellipses(point, ellipse_param) # indicator<0, if outside ellipse
    indicator = cs.fmax(0.0, indicator)**2 * alpha
    cost = cs.sum1(indicator * weight)
    # narrowness = 5
    # cost = cs.sum1( weight / (1+cs.exp(-narrowness*indicator-4)) * alpha )
    return cost

def cost_control_action(action:cs.MX, weight:cs.MX):
    cost = cs.sum1(weight*action**2)
    return cost

def cost_control_jerk(action:cs.MX, last_action:cs.MX, weight:cs.MX):
    cost = cs.sum1(weight*(action-last_action)**2)
    return cost

def cost_fleet_collision(point:cs.MX, points:cs.MX, safe_distance:float, weight:float):
    #cost for colliding with other robots
    cost = weight * cs.sum2(cs.fmax(0.0, safe_distance**2 - dist_to_points_square(point, points)))
    return cost

def cost_refvalue_deviation(actual_value:cs.MX, ref_value:cs.MX, weight=1):
    return weight*(actual_value-ref_value)**2

def cost_refstate_deviation(state:cs.MX, ref_state:cs.MX, weights:cs.MX):
    return (state-ref_state)**2 * weights

def cost_refpath_deviation(point:Union[cs.MX, cs.DM], line_segments:List[Union[cs.MX, cs.DM]], weight:float=1):
    '''
    Description:
        [Cost] Reference deviation error, penalizes on the deviation from the reference path.
    Arguments:
        line_segments - from the the start point to the end point
    Comments:
        The 'line_segments' contains segments which are end-to-end.
    '''
    distances_sqrt = cs.MX.ones(1)
    for i in range(len(line_segments)-1):
        distance = dist_to_lineseg(point, [line_segments[i], line_segments[i+1]])
        distances_sqrt = cs.horzcat(distances_sqrt, distance**2)
    cost = cs.mmin(distances_sqrt[1:]) * weight
    return cost

def cost_refpoint_detach(point:Union[cs.MX, cs.DM], ref_point:Union[cs.MX, cs.DM], ref_distance:float, weight:float=1):
    # The robot should stay a constant distance with some leader
    actual_distance = cs.sqrt(cs.sum1((point-ref_point)**2))
    cost = (actual_distance - ref_distance)**2 * weight
    return cost

class MyCallback(cs.Callback):
  def __init__(self, name, d, opts={}):
    cs.Callback.__init__(self)
    self.d = d
    self.construct(name, opts)

  # Number of inputs and outputs
  def get_n_in(self): return 1
  def get_n_out(self): return 1

  # Initialize the object
  def init(self):
     print('initializing object')

  # Evaluate numerically
  def eval(self, arg):
    x = 0
    if isinstance(arg[0], (cs.MX, cs.DM, cs.MX)):
        x = arg[0].full()  # Evaluate to numpy array
        print("Symbolic value is " + str(x))
    else:
       x = arg[0]
       print("Numerical value is " + str(x))
    f = cs.sin(self.d*x)
    return [f]

#%%## Main class ###
class MpcModule:
    '''
    Description:
        Build the MPC module. Define states, inputs, cost, and constraints.
    Arguments:
        config  <Configurator> - Contains all information/parameters needed.
    Attributes:
        print_name    <str>     - The name to print while running this class.
        config        <dotdict> - As above mentioned.
    Functions
        build              <pre>  - Build the MPC problem and solver.
    '''
    def __init__(self, config:Configurator):
        self.__print_name = '[MPC]'
        self.config = config
        # Frequently used
        self.ts = self.config.ts        # sampling time
        self.ns = self.config.ns        # number of states
        self.nu = self.config.nu        # number of inputs
        self.N_hor = self.config.N_hor  # control/pred horizon
        self.env = None
    
    def Generate_Lookup_Table(self, env:TrajectoryPlannerEnvironment = None, model:PerDDPG = None):
        if env != None:
            env = copy.deepcopy(env)
            boundary_array = env.boundary.get_padded_vertices()
            state1_range = np.linspace(boundary_array[3][0]
                                    , boundary_array[0][0]
                                    , 5) 
            state2_range = np.linspace(boundary_array[3][1]
                                    , boundary_array[1][1]
                                    , 5)
            state3_range = np.linspace(0
                                    , 2 * np.pi
                                    , 5)
            action1_range = np.linspace(self.config.lin_vel_min
                                    , self.config.lin_vel_max
                                    , 5)
            action2_range = np.linspace(-self.config.ang_vel_max
                                    , self.config.ang_vel_max
                                    , 5)
            timestep_range = np.linspace(0, 200, 5)
        else:
            state1_range = np.linspace(-1, 1, 5) 
            state2_range = np.linspace(-1, 1, 5)  # State 2 range
            state3_range = np.linspace(-1, 1, 5)  # State 3 range
            action1_range = np.linspace(-1, 1, 5)  # Action 1 range
            action2_range = np.linspace(-1, 1, 5)  # Action 2 range
            timestep_range = np.linspace(0, 200, 5)

        state1_range = state1_range.tolist()
        state2_range = state2_range.tolist()
        state3_range = state3_range.tolist()
        action1_range = action1_range.tolist()
        action2_range = action2_range.tolist()

        # Generate all combinations of state and action
        state_combinations = list(product(state1_range, state2_range, state3_range))  # 1000 combinations
        action_combinations = list(product(action1_range, action2_range))  # 100 combinations

        # Loop through all states and actions
        state_tensor = torch.tensor(state_combinations, dtype=torch.float32)
        action_tensor = torch.tensor(action_combinations, dtype=torch.float32)

        q_values = []
        for time in timestep_range:
            env.step_obstacles()
            for state in state_tensor:
                for action in action_tensor:
                    # Set the environment state and action
                    env.set_agent_state(state[:2], state[2], action[0], action[1])
                    env.update_status(reset=False)  # Update the environment state
                    obsv = env.get_observation()   # Get the observation
                    done = env.update_termination()  # Update termination (though not used here)

                    # Ensure action is a tensor
                    obsv = {key: torch.tensor(value, dtype=torch.float32).unsqueeze(0) for key, value in obsv.items()}
                    actions = action.unsqueeze(0) # Shape: (1, 2)

                    # Compute Q-value for the observation-action pair
                    q_value = model.critic.q1_forward(obsv, actions)  # Add batch dimension
                    q_values.append(q_value.item())  # Append the scalar Q-value to the list
            

        # Flatten the Q-values for the lookup table
        q_values_flat = np.array(q_values).flatten()  # Shape: (100000,)

        lookupTable = cs.interpolant(
            "q_lookup",
            "bspline",
            [state1_range, state2_range
             , state3_range, action1_range
             , action2_range, timestep_range],
            q_values_flat
        )
        self.lookupTable = lookupTable
        self.q_max = np.max(q_values_flat)

    def build(self, dynamics: Callable[[cs.MX, cs.MX, float], cs.MX], use_tcp:bool=False
              , isConsiderDRL = False, env:TrajectoryPlannerEnvironment = None
              , model:PerDDPG = None):
        """Build the MPC problem and solver, including states, inputs, cost, and constraints.

        Args:
            dynamics: Callable function that generates next state given the current state and action.
            use_tcp : If the solver will be called directly or via TCP.
        Conmments:
            Inputs (u): speed, angular speed
            states (s): x, y, theta, e (e is the channel width / allowable divation from reference, not included yet)
            Constraints (z):    1. Initialization, states, inputs
                                2. Penalty weights: qp, qv, qtheta, rv, rw; qN, qthetaN, qCTE, acc_penalty, omega_acc_penalty
                                3. Reference path and speed reference in each step (N_hor)
                                4. Other robots
                                5. static and dynamic obstacles
        Reference:
            Ellipse definition: [https://math.stackexchange.com/questions/426150/what-is-the-general-equation-of-the-ellipse-that-is-not-in-the-origin-and-rotate]
        """
        print(f'{self.__print_name} Building MPC module...')

        u = cs.MX.sym('u', self.nu*self.N_hor)              # 1. Inputs at every predictive step
        s = cs.MX.sym('s', 2*self.ns + self.nu)             # 2. Current and goal states + initial inputs
        q = cs.MX.sym('q', self.config.nq)                  # 3. Penalty parameters
        r = cs.MX.sym('r', self.ns*self.N_hor + self.N_hor) # 4. Reference path + speed reference in each step
        c = cs.MX.sym('c', self.ns*self.N_hor*self.config.Nother)                   # 5. Predicted states of other robots
        o_s = cs.MX.sym('os', self.config.Nstcobs*self.config.nstcobs)              # 6. Static obstacles
        o_d = cs.MX.sym('od', self.config.Ndynobs*self.config.ndynobs*self.N_hor)   # 7. Dynamic obstacles
        q_stc = cs.MX.sym('qstc', self.N_hor)               # 8. Static obstacle weights
        q_dyn = cs.MX.sym('qdyn', self.N_hor)               # 9. Dynamic obstacle weights
        horizon = cs.MX.sym('h',1)
        cur_timestep = cs.MX.sym('tm',1)
        q_qval = cs.MX.sym('qqval',1)
        z = cs.vertcat(s, q, r, c, o_s, o_d, q_stc, q_dyn, horizon, cur_timestep,q_qval)

        
        (x, y, theta, x_goal, y_goal, theta_goal, v_init, w_init) = (s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7])
        (qpos, qvel, qtheta, rv, rw)                    = (q[0], q[1], q[2], q[3], q[4])
        (qN, qthetaN, qrpd, acc_penalty, w_acc_penalty) = (q[5], q[6], q[7], q[8], q[9])
        
        path_ref = [cs.vertcat(r[i*self.ns], r[i*self.ns+1]) for i in range(self.N_hor)]
        path_ref.append(path_ref[-1])

        cost = 0
        penalty_constraints = 0
        state_next = cs.vcat([x,y,theta])

        if isConsiderDRL:
            self.Generate_Lookup_Table(env=env,model=model)

        for kt in range(0, self.N_hor): # LOOP OVER TIME STEPS
            
            ### Run step with motion model
            u_t = u[kt*self.nu:(kt+1)*self.nu]  # inputs at time t
            state_next = dynamics(state_next, u_t, self.ts) # Kinematic/dynamic model

            ### Reference deviation costs
            cost += cost_refpath_deviation(state_next, path_ref[kt:], weight=qrpd) # [cost] reference path deviation cost
            cost += cost_refvalue_deviation(u_t[0], r[self.ns*self.N_hor+kt], weight=qvel) # [cost] refenrence velocity deviation
            cost += cost_control_action(u_t, cs.vertcat(rv, rw)) # [cost] penalize control actions

            ### Fleet collision avoidance
            other_robots_x = c[kt*self.ns  ::self.ns*self.N_hor] # first  state
            other_robots_y = c[kt*self.ns+1::self.ns*self.N_hor] # second state
            other_robots = cs.hcat([other_robots_x, other_robots_y]) # states of other robots at time kt (Nother*ns)
            other_robots = cs.transpose(other_robots) # every column is a state of a robot
            cost += cost_fleet_collision(state_next[:2], other_robots, safe_distance=self.config.vehicle_width, weight=1000)

            ### Static obstacles
            for i in range(self.config.Nstcobs):
                eq_param = o_s[i*self.config.nstcobs : (i+1)*self.config.nstcobs]
                n_edges = int(self.config.nstcobs / 3) # 3 means b, a0, a1
                b, a0, a1 = eq_param[:n_edges], eq_param[n_edges:2*n_edges], eq_param[2*n_edges:]

                inside_stc_obstacle = inside_pollygon(state_next, b, a0, a1)
                penalty_constraints += cs.fmax(0, cs.vertcat(inside_stc_obstacle))

                # cost += cost_inside_polygon(state_next, b, a0, a1, weight=q_stc[kt])

            ### Dynamic obstacles
            # (x, y, rx, ry, tilted_angle, alpha) for obstacle 0 for N_hor steps, then similar for obstalce 1 for N_hor steps...
            x_dyn     = o_d[kt*self.config.ndynobs  ::self.config.ndynobs*self.N_hor]
            y_dyn     = o_d[kt*self.config.ndynobs+1::self.config.ndynobs*self.N_hor]
            rx_dyn    = o_d[kt*self.config.ndynobs+2::self.config.ndynobs*self.N_hor]
            ry_dyn    = o_d[kt*self.config.ndynobs+3::self.config.ndynobs*self.N_hor]
            As        = o_d[kt*self.config.ndynobs+4::self.config.ndynobs*self.N_hor]
            alpha_dyn = o_d[kt*self.config.ndynobs+5::self.config.ndynobs*self.N_hor]

            inside_dyn_obstacle = inside_ellipses(state_next, [x_dyn, y_dyn, rx_dyn, ry_dyn, As])
            penalty_constraints += cs.fmax(0, inside_dyn_obstacle)

            cost += cost_inside_ellipses(state_next, [x_dyn, y_dyn, rx_dyn+self.config.social_margin, ry_dyn+self.config.social_margin, As, alpha_dyn], weight=q_dyn[kt])

            # Also consider each element on the horizon for the q values
            if isConsiderDRL:
                inputs = cs.vertcat(state_next[0], state_next[1], state_next[2]
                                    , u_t[0], u_t[1], cur_timestep)
                cost += q_qval * (self.q_max - self.lookupTable(inputs))**2
        
        ### Terminal cost
        # state_final_goal = cs.vertcat(x_goal, y_goal, theta_goal)
        # cost += cost_refstate_deviation(state_next, state_final_goal, weights=cs.vertcat(qN, qN, qthetaN)) 
        if isConsiderDRL == False:
            cost += qN*((state_next[0]-x_goal)**2 + (state_next[1]-y_goal)**2) + qthetaN*(state_next[2]-theta_goal)**2 # terminated cost
        else:
            inputs = cs.vertcat(state_next[0], state_next[1], state_next[2]
                                , u_t[0], u_t[1], cur_timestep)
            cost += q_qval * (self.q_max - self.lookupTable(inputs))**2

        ### Max speed bound
        umin = [self.config.lin_vel_min, -self.config.ang_vel_max] * self.N_hor
        umax = [self.config.lin_vel_max,  self.config.ang_vel_max] * self.N_hor
        bounds = og.constraints.Rectangle(umin, umax)

        ### Acceleration bounds and cost
        v = u[0::2] # velocity
        w = u[1::2] # angular velocity
        acc   = (v-cs.vertcat(v_init, v[0:-1]))/self.ts
        w_acc = (w-cs.vertcat(w_init, w[0:-1]))/self.ts
        acc_constraints = cs.vertcat(acc, w_acc)
        # Acceleration bounds
        acc_min   = [ self.config.lin_acc_min] * self.N_hor 
        w_acc_min = [-self.config.ang_acc_max] * self.N_hor
        acc_max   = [ self.config.lin_acc_max] * self.N_hor
        w_acc_max = [ self.config.ang_acc_max] * self.N_hor
        acc_bounds = og.constraints.Rectangle(acc_min + w_acc_min, acc_max + w_acc_max)
        # Accelerations cost
        cost += cs.mtimes(acc.T, acc)*acc_penalty
        cost += cs.mtimes(w_acc.T, w_acc)*w_acc_penalty

        problem = og.builder.Problem(u, z, cost) \
            .with_constraints(bounds) \
            .with_aug_lagrangian_constraints(acc_constraints, acc_bounds)
        problem.with_penalty_constraints(penalty_constraints)

        build_config = og.config.BuildConfiguration() \
            .with_build_directory(self.config.build_directory) \
            .with_build_mode(self.config.build_type)
        if not use_tcp:
            build_config.with_build_python_bindings()
        else:
            build_config.with_tcp_interface_config()

        meta = og.config.OptimizerMeta() \
            .with_optimizer_name(self.config.optimizer_name)

        solver_config = og.config.SolverConfiguration() \
            .with_initial_penalty(10) \
            .with_max_duration_micros(MAX_SOVLER_TIME)
            # initial penalty = 1
            # tolerance = 1e-4
            # max_inner_iterations = 500 (given a penalty factor)
            # max_outer_iterations = 10  (increase the penalty factor)
            # penalty_weight_update_factor = 5.0
            # max_duration_micros = 5_000_000 (5 sec)

        builder = og.builder.OpEnOptimizerBuilder(problem, meta, build_config, solver_config) \
            .with_verbosity_level(1)
        builder.build()

        print(f'{self.__print_name} MPC module built.')

