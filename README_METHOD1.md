# Method 1: Running MPC with Q-Function Integration

### Link to Repository
https://github.com/Mathanesh/Design-Project-14.2/tree/main

## Setup and Execution

1. Use Main branch

2. Activate the correct conda environment:
```bash
conda activate drl-traj-plan
```

3. Modify main_continous.py per needs. To run MPC with Q function integration choose decision mode. Other variables can also be configured.

4. Run the script:
```bash
python main_continous.py
```

## Configuration Options

You can modify these parameters in main_continous.py:

### Scene Options
* First number (1 or 2): Scene type
* Second number (1-4): Obstacle configuration
* Third number (1-4): Specific variation

### Decision Modes
* 0: Pure MPC with Q-function
* 1: Pure DDPG
* 2: Pure TD3
* 3: Hybrid DDPG
* 4: Hybrid TD3

## Visualization

Set `to_plot=True` in the main function call to visualize the robot's trajectory.

The visualization shows:
* Robot position and orientation
* Obstacle configurations
* Reference path
* Actual trajectory
* Speed and angular velocity plots
