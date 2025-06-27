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

## Quick Demonstration (2-Hour Cap)

This demonstration showcases all key implementations within 2 hours as required for assignment evaluation.

```bash
# Step 1: Test individual agents on Assignment 2 main grid (15 minutes)
python train.py grid_configs/A2/assignment2_main.npy --agent_type random --episodes 5 --no_gui
python train.py grid_configs/A2/assignment2_main.npy --agent_type heuristic --episodes 5 --no_gui
python train.py grid_configs/A2/assignment2_main.npy --agent_type dqn --episodes 15 --no_gui
python train.py grid_configs/A2/assignment2_main.npy --agent_type ppo --episodes 15 --no_gui

# Step 2: Algorithm comparison (30 minutes)
python experimental_framework/algorithm_comparison.py --quick

# Step 3: Hyperparameter optimization demo (25 minutes)
python experimental_framework/hyperparameter_tuner.py --quick --max_configs 5 --episodes 15

# Step 4: Architecture ablation (20 minutes)
python experimental_framework/ablation_studies.py --quick --episodes 15

# Step 5: Evaluation framework demo (15 minutes)
python experimental_framework/evaluation_framework.py --quick

# Step 6: Generate visualizations (10 minutes)
python experimental_framework/learning_curves_generator.py
python experimental_framework/network_vis.py
python experimental_framework/cross_environment_performance.py

# Expected results: DQN variants outperform PPO, statistical significance confirmed
```

**For comprehensive agent testing across all environments**, run individual combinations as needed:
```bash
# Test specific agent-environment combinations
python train.py grid_configs/A2/open_space.npy --agent_type dqn --episodes 20 --no_gui
python train.py grid_configs/A2/simple_restaurant.npy --agent_type ppo --episodes 25 --no_gui
python train.py grid_configs/A2/corridor_test.npy --agent_type heuristic --episodes 10 --no_gui
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
- **PPO Agent**: Proximal Policy Optimization
- **Double DQN**: Improved Q-learning with overestimation bias reduction
- **Dueling DQN**: Value-advantage decomposition architecture

### **Restaurant Grids**
- `open_space.npy` - Basic testing environment
- `simple_restaurant.npy` - Core restaurant layout
- `corridor_test.npy` - Navigation challenges
- `maze_challenge.npy` - Complex pathfinding
- `assignment2_main.npy` - Comprehensive evaluation

## Usage

### **Basic Training**
```bash
python train.py <GRID> [options]

Required:
  GRID                    Path to grid file (.npy)

Agent Options:
  --agent_type {random,heuristic,dqn,ppo}  Agent to train (default: heuristic)
  --episodes N            Number of episodes (default: 100)
  --agent_start_pos X Y   Starting position (recommended for consistent results)

Environment Options:
  --no_gui               Disable visual interface (faster training)
  --sigma FLOAT          Environment stochasticity (default: 0.1)
  --random_seed INT      Random seed (default: 0)

Training Options:
  --save_agent PATH      Save trained model
  --load_agent PATH      Load pre-trained model
  --iter INT             Max steps per episode (default: 1000)
```

### **Example Commands**

#### **Development Testing**
```bash
# Quick validation
python train.py grid_configs/A2/open_space.npy --agent_type heuristic --episodes 5

# DQN training on simple grid
python train.py grid_configs/A2/simple_restaurant.npy --agent_type dqn --episodes 30 --no_gui

# PPO training
python train.py grid_configs/A2/assignment2_main.npy --agent_type ppo --episodes 50 --no_gui

# Performance comparison
python train.py grid_configs/A2/assignment2_main.npy --agent_type heuristic --episodes 20 --agent_start_pos 3 9
python train.py grid_configs/A2/assignment2_main.npy --agent_type dqn --episodes 50 --agent_start_pos 3 9
```

#### **Model Management**
```bash
# Train and save DQN model
python train.py grid_configs/A2/assignment2_main.npy --agent_type dqn --episodes 100 --save_agent "my_model.pth" --no_gui

# Load and continue training
python train.py grid_configs/A2/assignment2_main.npy --agent_type dqn --episodes 50 --load_agent "my_model.pth" --no_gui

# Evaluate saved model
python train.py grid_configs/A2/assignment2_main.npy --agent_type dqn --episodes 1 --load_agent "my_model.pth" --agent_start_pos 3 9
```

## Experimental Framework

### **Individual Experiment Scripts**

#### **Algorithm Comparison**
```bash
# Compare all algorithms with statistical analysis
python experimental_framework/algorithm_comparison.py

# Options:
# --quick              Reduced parameters for quick testing
# --runs N             Number of independent runs (default: 5)
# --episodes N         Episodes per run (default: 100)
# --environments LIST  Specific environments to test
```

#### **Hyperparameter Optimization**
```bash
# DQN hyperparameter tuning
python experimental_framework/hyperparameter_tuner.py

# Options:
# --quick              Quick mode with fewer configurations
# --max_configs N      Maximum configurations to test
# --episodes N         Episodes per configuration
# --method {random,grid}  Search strategy

# PPO environment-specific optimization
python experimental_framework/ppo_hyperparameter_optimizer.py

# Options:
# --quick              Quick mode
# --episodes N         Episodes per configuration
# --environments LIST  Environments to optimize for
```

#### **Ablation Studies**
```bash
# Network architecture ablation
python experimental_framework/ablation_studies.py

# Options:
# --quick              Quick mode with fewer architectures
# --episodes N         Episodes per architecture
# --study {architecture,components}  Type of ablation study
```

#### **Evaluation Framework**
```bash
# Comprehensive evaluation with advanced metrics
python experimental_framework/evaluation_framework.py

# Options:
# --quick              Quick mode
# --episodes N         Episodes for evaluation
# --metrics LIST       Specific metrics to compute
# --statistical_tests  Run statistical significance tests
```

### **Visualization Scripts**

#### **Reproduce Report Figures**
```bash
# Generate learning curves (Figure 2)
python experimental_framework/learning_curves_generator.py

# Generate network architecture comparison (Figure 3)
python experimental_framework/network_vis.py

# Generate cross-environment performance comparison (Figure 1)
python experimental_framework/cross_environment_performance.py
```

**Note**: Visualization scripts read data from experiment outputs. Run the corresponding experiment scripts first:
- `learning_curves_generator.py` reads from `algorithm_comparison.py` output
- `network_vis.py` reads from `ablation_studies.py` output  
- `cross_environment_performance.py` reads from `evaluation_framework.py` output

## Project Structure

```
├── agents/
│   ├── base_agent.py            # Abstract base class for all agents
│   ├── random_agent.py          # Random baseline agent
│   ├── heuristic_agent.py       # Rule-based intelligent agent
│   ├── DQN_agent.py             # Deep Q-Network implementation
│   ├── DQN_nn.py                # Neural network architectures
│   └── PPO_agent.py             # PPO implementation
├── world/
│   ├── environment.py           # Main environment with 8D state space
│   ├── grid.py                  # Grid representation
│   ├── gui.py                   # Visualization interface
│   ├── helpers.py               # Utility functions
│   └── create_restaurant_grids.py  # Grid generation
├── experimental_framework/      # Individual experimental scripts
│   ├── algorithm_comparison.py  # Multi-algorithm comparison
│   ├── hyperparameter_tuner.py  # DQN hyperparameter optimization
│   ├── ppo_hyperparameter_optimizer.py # PPO-specific tuning
│   ├── ablation_studies.py      # Component importance analysis
│   ├── evaluation_framework.py  # Advanced RL metrics
│   ├── learning_curves_generator.py    # Generate Figure 2
│   ├── network_vis.py           # Generate Figure 3
│   └── cross_environment_performance.py # Generate Figure 1
├── grid_configs/
│   ├── A1_grid.npy              # Main assignment grid
│   └── A2/                      # Assignment 2 specific grids
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

## Reproducing Paper Results

### **Complete Experimental Pipeline**
```bash
# Step 1: Run all experiments (generates data)
python experimental_framework/algorithm_comparison.py --runs 5 --episodes 100
python experimental_framework/ablation_studies.py --episodes 100
python experimental_framework/evaluation_framework.py --episodes 100 --statistical_tests

# Step 2: Generate all report figures
python experimental_framework/learning_curves_generator.py    # Figure 2
python experimental_framework/network_vis.py                 # Figure 3  
python experimental_framework/cross_environment_performance.py # Figure 1

# Step 3: Run hyperparameter optimization (for tables)
python experimental_framework/hyperparameter_tuner.py
python experimental_framework/ppo_hyperparameter_optimizer.py
```

### **Individual Result Reproduction**
```bash
# Table 2: PPO environment-specific configurations
python experimental_framework/ppo_hyperparameter_optimizer.py --all_environments

# Table 3: ANOVA statistical analysis  
python experimental_framework/evaluation_framework.py --statistical_tests

# Table 4: Cross-environment performance comparison
python experimental_framework/algorithm_comparison.py --runs 5

# Table 5: Architecture ablation results
python experimental_framework/ablation_studies.py

# Figures 2-3: Learning curves and architecture comparison
python experimental_framework/learning_curves_generator.py
python experimental_framework/network_vis.py
python experimental_framework/cross_environment_performance.py
```

## Expected Performance

### **Performance Benchmarks** (with consistent starting positions)
- **Random Agent**: -1200 to -1600 avg reward, 10-30% success rate
- **Heuristic Agent**: -300 to -500 avg reward, 70-90% success rate
- **DQN Agent**: -400 to -900 avg reward, 60-85% success rate (varies by grid)
- **Double DQN**: -200 to -600 avg reward, 70-90% success rate
- **Dueling DQN**: -250 to -650 avg reward, 65-85% success rate
- **PPO Agent**: -800 to -1400 avg reward, 25-60% success rate (environment dependent)

### **Grid Difficulty Ranking**
1. **open_space** (Easiest) - All agents >95% success
2. **simple_restaurant** - DQN variants >90%, PPO ~55%
3. **corridor_test** - DQN variants >85%, PPO ~75%
4. **A1_grid** - DQN variants >70%, PPO ~45%
5. **assignment2_main** - DQN variants >60%, PPO ~25%
6. **maze_challenge** (Hardest) - DQN variants >55%, PPO ~20%

## Key Insights

### **Algorithm Performance**
- **DQN variants consistently outperform PPO** in discrete navigation tasks
- **Double DQN shows best overall performance** with overestimation bias reduction
- **Dueling DQN excels in complex environments** requiring value-advantage decomposition
- **PPO struggles with spatial reasoning** and high-dimensional discrete action spaces

### **Architecture Sensitivity**
- **DQN optimal**: 64-128 hidden units, 2-3 layers
- **PPO optimal**: 256 hidden units, 3 layers (requires more capacity)
- **Architectural choice significantly impacts performance** (15-25% variation)

### **Environment Complexity Impact**
- **Performance degrades with environment complexity** for all algorithms
- **PPO shows highest sensitivity** to environment difficulty
- **Starting position significantly affects learning** (position-dependent policies)

### **Hyperparameter Importance**
- **Learning rate most critical** for both algorithms
- **Entropy coefficient crucial for PPO** (exploration-exploitation balance)
- **Experience replay buffer size matters for DQN** (25k-50k optimal)

## Troubleshooting

### **Common Issues**
```bash
# Import errors
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Missing grids  
python world/create_restaurant_grids.py

# DQN performance issues
python train.py grid_configs/A1_grid.npy --agent_type dqn --episodes 100 --agent_start_pos 3 11  # Use fixed start

# PPO convergence problems
python experimental_framework/ppo_hyperparameter_optimizer.py  # Environment-specific tuning

# Compare with baseline
python train.py grid_configs/A1_grid.npy --agent_type heuristic --episodes 20 --agent_start_pos 3 11
```

### **Experimental Framework Issues**
```bash
# Quick validation of all agents
python experimental_framework/test_all_agents.py --episodes 5

# Memory issues with large experiments
python experimental_framework/algorithm_comparison.py --quick  # Reduced parameters

# Missing experimental dependencies
pip install -r requirements.txt
```

### **Visualization Issues**
```bash
# If visualization scripts fail, ensure experiments are run first
python experimental_framework/algorithm_comparison.py
python experimental_framework/ablation_studies.py  
python experimental_framework/evaluation_framework.py

# Then generate visualizations
python experimental_framework/learning_curves_generator.py
python experimental_framework/network_vis.py
python experimental_framework/cross_environment_performance.py
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