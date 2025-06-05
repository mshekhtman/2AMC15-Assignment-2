# Data Intelligence Challenge 2AMC15 - Assignment 2
## Restaurant Delivery Robot with Continuous State Space

This repository implements a **10D continuous state space environment** for training deep reinforcement learning agents on restaurant food delivery tasks.

## Assignment Focus

**Stakeholder**: Restaurant chain seeking to automate food delivery  
**Problem**: Train a delivery robot to efficiently navigate and deliver orders  
**Solution**: Deep RL with 10D continuous state representation

## Key Changes from Assignment 1

### **Environment Enhancements**
- **10D Continuous State Space**: Vector replacing discrete grid positions
- **Restaurant-Optimized Rewards**: Efficiency, safety, and goal achievement focus
- **Local Environment Sensing**: 4-directional clearance detection
- **Mission Progress Tracking**: Target counting and completion progress

### **State Representation (10D Vector)**
```python
[0-1]   : Normalized position (x, y)           # [0, 1]
[2-3]   : Direction to target (unit vector)    # [-1, 1]
[4]     : Remaining targets (normalized)       # [0, 1]
[5-8]   : Clear directions (front/left/right/back) # {0, 1}
[9]     : Mission progress                     # [0, 1]
```

## Quick Start

### Setup Environment
```bash
# Create conda environment
conda create -n dic2025 python=3.11
conda activate dic2025

# Install dependencies
pip install -r requirements.txt

# Create restaurant grids
python create_restaurant_grids.py
```

## Testing Guide

### **1. Create Test Grids**
```bash
python create_restaurant_grids.py
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

# Performance benchmark
python train.py grid_configs/A2/assignment2_main.npy --agent_type heuristic --episodes 25 --no_gui
```

### **3. DQN Agent Testing**

#### Quick DQN Validation
```bash
# Simple grid (fast training)
python train.py grid_configs/open_space.npy --agent_type dqn --episodes 50 --no_gui

# Restaurant layout
python train.py grid_configs/A2/simple_restaurant.npy --agent_type dqn --episodes 100 --no_gui
```

#### Full DQN Training
```bash
# Main assignment grid
python train.py grid_configs/A2/assignment2_main.npy --agent_type dqn --episodes 200 --no_gui --agent_start_pos 3 9
```

### **4. Performance Comparison**
```bash
# Compare all agents on same grid
python train.py grid_configs/A2/assignment2_main.npy --agent_type random --episodes 20 --no_gui
python train.py grid_configs/A2/assignment2_main.npy --agent_type heuristic --episodes 20 --no_gui  
python train.py grid_configs/A2/assignment2_main.npy --agent_type dqn --episodes 100 --no_gui
```

### **5. Visual Testing (with GUI)**
```bash
# Watch agents learn (slower)
python train.py grid_configs/A2/simple_restaurant.npy --agent_type heuristic --episodes 5 --fps 10
python train.py grid_configs/A2/simple_restaurant.npy --agent_type dqn --episodes 20 --fps 10
```

### **6. Grid Complexity Testing**
```bash
# Easy to hard progression
python train.py grid_configs/A2/open_space.npy --agent_type dqn --episodes 30 --no_gui
python train.py grid_configs/A2/simple_restaurant.npy --agent_type dqn --episodes 50 --no_gui
python train.py grid_configs/A2/corridor_test.npy --agent_type dqn --episodes 75 --no_gui
python train.py grid_configs/A2/maze_challenge.npy --agent_type dqn --episodes 100 --no_gui
```

## File Structure

```
├── agents/
│   ├── base_agent.py            # Enhanced for 10D continuous states
│   ├── random_agent.py          # Random baseline
│   ├── heuristic_agent.py       # Intelligent rule-based agent
│   ├── DQN_agent.py             # DQN implementation
│   └── DQN_nn.py                # Neural network architecture
├── world/
│   ├── environment.py          # 10D continuous state implementation
│   ├── grid.py                 # Grid representation
│   ├── gui.py                  # Visualization
│   ├── helpers.py              # Utilities and path visualization
│   └──create_restaurant_grids.py  # Grid generation script
├── grid_configs/
│   ├── open_space.npy          # Basic testing
│   ├── simple_restaurant.npy   # Core restaurant layout
│   ├── corridor_test.npy       # Navigation testing
│   ├── maze_challenge.npy      # Complex navigation
│   └── assignment2_main.npy    # Primary evaluation grid
├── train.py                    # Training script
├── README.md                   # README file
└── requirements.txt            # Dependencies
```

## Available Agents

| Agent | Type | Use Case | Expected Performance |
|-------|------|----------|---------------------|
| `random` | Baseline | Sanity check | ~10% success |
| `heuristic` | Rule-based | Strong baseline | ~80-90% success |
| `dqn` | Deep RL | Assignment 2 baseline | ~95-100% success |

## Environment Configuration

```python
# 10D continuous state (default)
env = Environment(
    grid_fp="grid_configs/assignment2_main.npy",
    state_representation="continuous_vector",
    agent_start_pos=(3, 9)
)

# Backward compatible discrete
env = Environment(
    grid_fp="grid_configs/assignment2_main.npy", 
    state_representation="discrete"
)
```

## Expected Results

### Performance Benchmarks
- **Random Agent**: -500 to -800 average reward, 10-20% success rate
- **Heuristic Agent**: -50 to -150 average reward, 80-95% success rate  
- **DQN Agent**: -5 to -50 average reward, 95-100% success rate

### Training Progress (DQN)
- **Episodes 0-20**: Learning basic navigation (-500 to -200 reward)
- **Episodes 20-50**: Improving efficiency (-200 to -100 reward)
- **Episodes 50+**: Near-optimal performance (-50 to 0 reward)

## Output Files

After training, check:
- **`results/`**: Path visualizations and statistics
- **Training logs**: Episode rewards, success rates, learning progress
- **GUI**: Real-time agent movement (without `--no_gui`)

## Troubleshooting

### Common Issues
```bash
# Import errors
export PYTHONPATH="${PYTHONPATH}:$(pwd)"  # Linux/Mac
set PYTHONPATH=%PYTHONPATH%;%cd%          # Windows

# Missing grids
python create_restaurant_grids.py

# No path visualization
mkdir results
```

### Validation Test
```bash
# Quick environment check
python train.py grid_configs/open_space.npy --agent_type random --episodes 1
```