# Data Intelligence Challenge 2AMC15 - Assignment 2
## Restaurant Delivery Robot with Realistic Continuous State Space

This repository implements a **8D realistic continuous state space environment** for training deep reinforcement learning agents on restaurant food delivery tasks.

## Assignment Focus

**Stakeholder**: Restaurant chain seeking to automate food delivery  
**Problem**: Train a delivery robot to efficiently navigate and deliver orders  
**Solution**: Deep RL with 8D realistic state representation

## Key Changes from Assignment 1

### **Environment Enhancements**
- **8D Realistic Continuous State Space**: Vector based on actual robot sensors
- **Restaurant-Optimized Rewards**: Efficiency, safety, and goal achievement focus
- **Local Environment Sensing**: 4-directional obstacle detection (lidar/camera simulation)
- **Mission Progress Tracking**: Target counting and completion progress
- **No Omniscient Knowledge**: Agent must explore to find targets (realistic!)

### **Realistic State Representation (8D Vector)**
```python
[0-1]   : Normalized position (x, y)           # GPS/odometry [0, 1]
[2-5]   : Clear directions (down/up/left/right) # Sensor data {0, 1}
[6]     : Remaining targets (normalized)       # Mission tracking [0, 1]
[7]     : Mission progress                     # Internal status [0, 1]
```

### **Training Progress Tracking**
- **Logger System**: Tracks training vs evaluation performance
- **Dual Performance Metrics**: ε-greedy training vs greedy evaluation
- **Visual Learning Curves**: Automatic plot generation
- **Model Save/Load**: Continue training from checkpoints

## Quick Start

### Setup Environment
```bash
# Create conda environment
conda create -n dic2025 python=3.11
conda activate dic2025

# Install dependencies
pip install -r requirements.txt

# Create restaurant grids
python world/create_restaurant_grids.py
```

## Key Features

### **8D Realistic State Space**
Unlike traditional grid-based RL, our environment uses realistic robot sensors:
```python
[0-1]: Normalized position (x, y)     # GPS/odometry
[2-5]: Clear directions (UDLR)        # Lidar/camera sensors  
[6-7]: Mission status                 # Task management system
```

### **Available Agents**
- **Random Agent**: Baseline performance
- **Heuristic Agent**: Rule-based intelligent baseline
- **DQN Agent**: Deep Q-Network with 8D state input
- **PPO Agent**: Proximal Policy Optimization (coming soon)

### **Restaurant Grids**
- `A1_grid.npy` - Main assignment grid
- `open_space.npy` - Basic testing environment
- `simple_restaurant.npy` - Core restaurant layout
- `corridor_test.npy` - Navigation challenges
- `maze_challenge.npy` - Complex pathfinding
- `assignment2_main.npy` - Comprehensive evaluation

## Usage

### **Training Arguments**
```bash
python train.py <GRID> [options]

Required:
  GRID                    Path to grid file (.npy)

Agent Options:
  --agent_type {random,heuristic,dqn}  Agent to train (default: heuristic)
  --episodes N            Number of episodes (default: 100)
  --agent_start_pos X Y   Starting position (recommended for consistent results)

Environment Options:
  --no_gui               Disable visual interface (faster training)
  --sigma FLOAT          Environment stochasticity (default: 0.1)
  --random_seed INT      Random seed (default: 0)

Training Options:
  --save_agent PATH      Save trained DQN model
  --load_agent PATH      Load pre-trained DQN model
  --iter INT             Max steps per episode (default: 1000)
```

### **Example Commands**

#### **Development Testing**
```bash
# Quick validation
python train.py grid_configs/A2/open_space.npy --agent_type heuristic --episodes 5

# DQN training on simple grid
python train.py grid_configs/A2/simple_restaurant.npy --agent_type dqn --episodes 30 --no_gui

# Performance comparison
python train.py grid_configs/A1_grid.npy --agent_type heuristic --episodes 20 --agent_start_pos 3 11
python train.py grid_configs/A1_grid.npy --agent_type dqn --episodes 50 --agent_start_pos 3 11
```

#### **Model Management**
```bash
# Train and save DQN model
python train.py grid_configs/A1_grid.npy --agent_type dqn --episodes 100 --save_agent "my_model.pth" --no_gui

# Load and continue training
python train.py grid_configs/A1_grid.npy --agent_type dqn --episodes 50 --load_agent "my_model.pth" --no_gui

# Evaluate saved model
python train.py grid_configs/A1_grid.npy --agent_type dqn --episodes 1 --load_agent "my_model.pth" --agent_start_pos 3 11
```

#### **Grid Complexity Progression**
```bash
# Easy → Hard progression
python train.py grid_configs/A2/open_space.npy --agent_type dqn --episodes 20 --no_gui
python train.py grid_configs/A2/simple_restaurant.npy --agent_type dqn --episodes 40 --no_gui  
python train.py grid_configs/A1_grid.npy --agent_type dqn --episodes 60 --no_gui
python train.py grid_configs/A2/maze_challenge.npy --agent_type dqn --episodes 80 --no_gui
```

## Project Structure

```
├── agents/
│   ├── base_agent.py            # Abstract base class for all agents
│   ├── random_agent.py          # Random baseline agent
│   ├── heuristic_agent.py       # Rule-based intelligent agent
│   ├── DQN_agent.py             # Deep Q-Network implementation
│   ├── DQN_nn.py                # Neural network architectures
│   └── PPO_agent.py             # PPO implementation (coming soon)
├── world/
│   ├── environment.py           # Main environment with 8D state space
│   ├── grid.py                  # Grid representation
│   ├── gui.py                   # Visualization interface
│   ├── helpers.py               # Utility functions
│   └── create_restaurant_grids.py  # Grid generation
├── grid_configs/
│   ├── A1_grid.npy              # Main assignment grid
│   └── A2/                      # Assignment 2 specific grids
├── experimental_framework/      # Advanced analysis tools
├── train.py                     # Main training script
├── logger.py                    # Training progress tracking
└── README.md                    # This file
```

## Environment Details

### **State Representation**
Our 8D continuous state vector represents realistic robot sensing:

```python
def get_realistic_delivery_state(self) -> np.ndarray:
    """Generate 8D state vector from robot sensors."""
    return np.array([
        norm_x, norm_y,                    # Position (GPS/odometry)
        down_clear, up_clear,              # Movement clearance
        left_clear, right_clear,           # (lidar/camera sensors)
        remaining_targets_norm,            # Mission tracking
        progress                           # Task completion
    ])
```

### **Reward Function**
Enhanced reward function for better learning:
- **Target reached**: +30 (mission success)
- **Movement step**: -0.5 (efficiency incentive)
- **Collision**: -5 (safety penalty)
- **Progress bonus**: +5 × (distance improvement) (dense feedback)
- **Exploration bonus**: +0.5 (new area visited)

### **Action Space**
Discrete movement actions:
- `0`: Move down
- `1`: Move up  
- `2`: Move left
- `3`: Move right

## Expected Performance

### **Performance Benchmarks** (with consistent starting positions)
- **Random Agent**: -800 to -1200 avg reward, 10-20% success rate
- **Heuristic Agent**: -50 to -200 avg reward, 80-95% success rate
- **DQN Agent**: -200 to -800 avg reward, 50-80% success rate (varies by grid/position)

### **Training Progress** (DQN)
- **Episodes 0-20**: Random exploration, high negative rewards
- **Episodes 20-50**: Learning basic navigation patterns
- **Episodes 50+**: Policy refinement and optimization

## Output Files

Training automatically generates:
- **`results/`**: Training curves and performance plots
- **`*.pth`**: Saved DQN models (when using `--save_agent`)
- **Path visualizations**: Agent movement heatmaps
- **Performance logs**: Detailed training statistics

## Key Insights

### **Starting Position Dependency**
DQN performance varies significantly with starting position:
```bash
# Different starting positions can lead to 30-90% success rate variation
python train.py grid_configs/A1_grid.npy --agent_type dqn --episodes 50 --agent_start_pos 3 11  # Good
python train.py grid_configs/A1_grid.npy --agent_type dqn --episodes 50 --agent_start_pos 8 8   # Harder
```

### **Grid Difficulty Ranking**
1. **open_space** (Easiest) - Minimal obstacles
2. **simple_restaurant** - Basic restaurant layout  
3. **A1_grid** - Balanced complexity
4. **corridor_test** - Navigation challenges
5. **assignment2_main** - Complex restaurant
6. **maze_challenge** (Hardest) - Advanced pathfinding

### **Realistic Learning Challenge**
Without omniscient target knowledge:
- Agent must **explore to discover targets**
- **Local sensor data** limits long-range planning
- **Spatial memory** becomes crucial for navigation
- **Position-specific policies** don't generalize well

## Troubleshooting

### **Common Issues**
```bash
# Import errors
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Missing grids  
python world/create_restaurant_grids.py

# DQN performance issues
python train.py grid_configs/A1_grid.npy --agent_type dqn --episodes 100 --agent_start_pos 3 11  # Use fixed start

# Compare with baseline
python train.py grid_configs/A1_grid.npy --agent_type heuristic --episodes 20 --agent_start_pos 3 11
```

### **GPU/CUDA Setup**
```bash
# Test CUDA functionality
python test_cuda_setup.py

# Force CPU if needed
CUDA_VISIBLE_DEVICES="" python train.py grid_configs/A1_grid.npy --agent_type dqn
```

### **Quick Validation**
```bash
# Sanity check
python train.py grid_configs/A1_grid.npy --agent_type random --episodes 1 --agent_start_pos 3 11
```