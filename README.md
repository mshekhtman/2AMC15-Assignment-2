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
[2-5]   : Clear directions (front/left/right/back) # Sensor data {0, 1}
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

## Testing Guide

### **1. Create Test Grids**
```bash
python world/create_restaurant_grids.py
```
**Creates**: `open_space.npy`, `simple_restaurant.npy`, `corridor_test.npy`, `maze_challenge.npy`, `assignment2_main.npy`

### **2. Basic Agent Testing**

#### Random Agent (Baseline)
```bash
# Quick test
python train.py grid_configs/A2/open_space.npy --agent_type random --episodes 5

# Performance test  
python train.py grid_configs/A2/simple_restaurant.npy --agent_type random --episodes 20 --no_gui
```

#### Heuristic Agent (Smart Baseline)
```bash
# Visual test
python train.py grid_configs/A2/simple_restaurant.npy --agent_type heuristic --episodes 10

# Performance benchmark with fixed starting position
python train.py grid_configs/A1_grid.npy --agent_type heuristic --episodes 25 --no_gui --agent_start_pos 3 11
```

### **3. DQN Agent Testing**

#### Quick DQN Validation
```bash
# Simple grid (fast training)
python train.py grid_configs/A2/open_space.npy --agent_type dqn --episodes 50 --no_gui

# Restaurant layout
python train.py grid_configs/A2/simple_restaurant.npy --agent_type dqn --episodes 100 --no_gui
```

#### Full DQN Training with Save/Load
```bash
# Train and save model
python train.py grid_configs/A1_grid.npy --agent_type dqn --episodes 100 --no_gui --agent_start_pos 3 11 --save_agent "trained_model.pth"

# Load and continue training
python train.py grid_configs/A1_grid.npy --agent_type dqn --episodes 50 --no_gui --agent_start_pos 3 11 --load_agent "trained_model.pth"
```

### **4. Performance Comparison**
```bash
# Compare all agents on same grid with fixed starting position
python train.py grid_configs/A1_grid.npy --agent_type random --episodes 20 --no_gui --agent_start_pos 3 11
python train.py grid_configs/A1_grid.npy --agent_type heuristic --episodes 20 --no_gui --agent_start_pos 3 11
python train.py grid_configs/A1_grid.npy --agent_type dqn --episodes 100 --no_gui --agent_start_pos 3 11
```

### **5. Visual Training (with GUI)**
```bash
# Watch agents learn (slower)
python train.py grid_configs/A2/simple_restaurant.npy --agent_type heuristic --episodes 5 --fps 10
python train.py grid_configs/A1_grid.npy --agent_type dqn --episodes 20 --fps 10 --agent_start_pos 3 11
```

### **6. Grid Complexity Progression**
```bash
# Easy to hard progression with consistent starting positions
python train.py grid_configs/A2/open_space.npy --agent_type dqn --episodes 30 --no_gui --agent_start_pos 2 2
python train.py grid_configs/A2/simple_restaurant.npy --agent_type dqn --episodes 50 --no_gui --agent_start_pos 2 8
python train.py grid_configs/A2/corridor_test.npy --agent_type dqn --episodes 75 --no_gui --agent_start_pos 2 4
python train.py grid_configs/A2/maze_challenge.npy --agent_type dqn --episodes 100 --no_gui --agent_start_pos 2 6
```

## File Structure

```
├── agents/
│   ├── base_agent.py            # Enhanced for 8D realistic states
│   ├── random_agent.py          # Random baseline
│   ├── heuristic_agent.py       # Intelligent rule-based agent
│   ├── DQN_agent.py             # DQN implementation with save/load
│   └── DQN_nn.py                # Neural network architecture (8D input)
├── world/
│   ├── environment.py           # 8D realistic state implementation
│   ├── grid.py                  # Grid representation
│   ├── gui.py                   # Visualization
│   ├── helpers.py               # Utilities and path visualization
│   └── create_restaurant_grids.py  # Grid generation script
├── grid_configs/
│   ├── A1_grid.npy              # Main assignment grid
│   └── A2/                      # Additional test grids
│       ├── open_space.npy       # Basic testing
│       ├── simple_restaurant.npy # Core restaurant layout
│       ├── corridor_test.npy    # Navigation testing
│       ├── maze_challenge.npy   # Complex navigation
│       └── assignment2_main.npy # Primary evaluation grid
├── train.py                     # Training script with logger
├── logger.py                    # Training progress tracking
├── README.md                    # This file
└── requirements.txt             # Dependencies
```

## Available Agents

| Agent | Type | Use Case | Expected Performance |
|-------|------|----------|---------------------|
| `random` | Baseline | Sanity check | ~10-20% success |
| `heuristic` | Rule-based | Strong baseline | ~80-95% success |
| `dqn` | Deep RL | Assignment 2 focus | ~60-80% success* |

*Performance varies significantly with starting position due to exploration-based learning

## Environment Configuration

```python
# 8D realistic continuous state (default)
env = Environment(
    grid_fp="grid_configs/A1_grid.npy",
    state_representation="continuous_vector",
    agent_start_pos=(3, 11)  # Recommended: always specify starting position
)

# Backward compatible discrete state
env = Environment(
    grid_fp="grid_configs/A1_grid.npy", 
    state_representation="discrete"
)
```

## Training Arguments

```bash
python train.py <GRID> [options]

Required:
  GRID                    Path to grid file (.npy)

Optional:
  --agent_type {random,heuristic,dqn}  Agent to train (default: heuristic)
  --episodes N            Number of episodes (default: 100)
  --agent_start_pos X Y   Starting position (highly recommended)
  --no_gui               Disable visual interface (faster training)
  --save_agent PATH      Save trained DQN model
  --load_agent PATH      Load pre-trained DQN model
  --sigma FLOAT          Environment stochasticity (default: 0.1)
  --random_seed INT      Random seed (default: 0)
```

## Expected Results

### Performance Benchmarks (with fixed starting position)
- **Random Agent**: -800 to -1200 average reward, 10-20% success rate
- **Heuristic Agent**: -50 to -200 average reward, 80-95% success rate  
- **DQN Agent**: -500 to -1500 average reward, 60-80% success rate (position-dependent)

### Training Progress (DQN)
- **Episodes 0-20**: Random exploration (-2000+ average reward)
- **Episodes 20-50**: Learning basic navigation (-1500 to -1000 reward)
- **Episodes 50+**: Strategy refinement (-1000 to -500 reward)

### Logger Output
- **Training Plots**: `results/[timestamp]_targetrewardsplot.png`
- **Evaluation Plots**: `results/[timestamp]_DQNrewardsplot.png`
- **Performance Metrics**: Automatic summary statistics

## Output Files

After training, check:
- **`results/`**: Training curves and path visualizations
- **`*.pth`**: Saved DQN models (if `--save_agent` used)
- **Console**: Real-time progress and final statistics
- **GUI**: Real-time agent movement (without `--no_gui`)

## Key Insights from Testing

### **Starting Position Dependency**
DQN performance varies dramatically with starting position:
```bash
# Some positions lead to 90%+ success
python train.py grid_configs/A1_grid.npy --agent_type dqn --episodes 50 --agent_start_pos 3 11

# Other positions may lead to 30% success
python train.py grid_configs/A1_grid.npy --agent_type dqn --episodes 50 --agent_start_pos 8 8
```

### **Training vs Evaluation Gap**
Logger reveals important differences:
- **Training Success Rate**: 60-70% (includes exploration luck)
- **DQN Evaluation Rewards**: Often much worse (pure policy performance)
- **Final Evaluation**: Can vary from complete success to complete failure

### **Realistic Learning Challenge**
Without omniscient target knowledge:
- Agent must **explore to discover targets**
- **Spatial memory** becomes crucial
- **Local obstacle detection** limits long-range planning
- **Position-specific strategies** don't generalize well

## Troubleshooting

### Common Issues
```bash
# Import errors
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%cd%          # Windows

# Missing grids
python world/create_restaurant_grids.py

# Missing results directory
mkdir results

# Agent fails from different starting positions
# Always use --agent_start_pos for consistent testing
```

### Performance Issues
```bash
# If DQN performance is poor:
# 1. Use fixed starting position
python train.py grid_configs/A1_grid.npy --agent_type dqn --episodes 100 --agent_start_pos 3 11

# 2. Try longer training
python train.py grid_configs/A1_grid.npy --agent_type dqn --episodes 200 --agent_start_pos 3 11

# 3. Compare with heuristic baseline
python train.py grid_configs/A1_grid.npy --agent_type heuristic --episodes 20 --agent_start_pos 3 11
```

### Validation Test
```bash
# Quick environment check
python train.py grid_configs/A1_grid.npy --agent_type random --episodes 1 --agent_start_pos 3 11
```

## Research Notes

### State Space Design Philosophy
The 8D state space represents **realistic robot sensing**:
- **No omniscient target knowledge** (unlike unrealistic 10D version)
- **Local sensor simulation** (lidar/camera range)
- **Position tracking** (GPS/odometry equivalent)
- **Mission awareness** (task management system)

### Logger Functionality
The training logger provides crucial insights:
- **Separates exploration noise from true performance**
- **Tracks learning curves for both training and evaluation**
- **Reveals performance gaps and learning plateaus**
- **Enables hyperparameter optimization**

This creates a more challenging but realistic learning environment where agents must develop robust navigation strategies through exploration and spatial reasoning.