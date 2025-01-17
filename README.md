# Bird's-Eye-View Based Trajectory Planning of Multiple Mobile Robots Using DRL and MPC
*Trajectory Planning of Multiple Robots using Vision-Based Continuous Deep Reinforcement Learning and Model Predictive Control*
![Example](doc/cover.jpg "Example")

The main branch of this repository is the method 1 implementation. The method 2 and method 3 implementation can be found in the training-with-mpc branch in the same repository.

## Quick Start
### OpEn
The NMPC formulation is solved using open source implementation of PANOC, namely [OpEn](https://alphaville.github.io/optimization-engine/). Follow the [installation instructions](https://alphaville.github.io/optimization-engine/docs/installation) before proceeding. 

### Install dependencies (after installing OpEn)
```
pip install -r requirements.txt
```
or
```
conda env create -f environment.yaml
```
**NOTE** If you cannot create the virtual environment via conda, please create your own virtual environment (e.g. conda create -n rlboost python=3.9), and pip install.
Make sure your RUST is up-to-date and Pytorch is compatible with Cuda. 

### Generate MPC solver
Go to "test_block_mpc.py", change **INIT_BUILD** to true and run
```
python test_block_mpc.py
```
After this, a new directory *mpc_build* will appear and contain the solver. Then, you are good to go :)

### To train the DDPG
Go to "src/continous_training_local.py", change **load_checkpoint** to False and run.

## Use Case
Run *src/main_continous.py* for the simulation in Python. Several cases are available by changing ```scene_option``` in *src/main_continous.py*.


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
