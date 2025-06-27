import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def generate_learning_curves(json_file_path, output_path="learning_curves.png"):
    """
    Generate learning curves from comprehensive_results.json containing all algorithms
    
    Args:
        json_file_path: Path to the comprehensive_results.json file
        output_path: Where to save the generated plot
    """
    
    # Load the comprehensive data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Set up the plot with professional styling
    plt.style.use('default')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define colors for different algorithms
    colors = {
        'Random': '#e74c3c',      # Red
        'DQN': '#3498db',         # Blue  
        'Double DQN': '#2ecc71',  # Green
        'Dueling DQN': '#f39c12', # Orange
        'PPO': '#9b59b6'          # Purple
    }
    
    # Plot learning curves for each algorithm
    for algorithm, color in colors.items():
        if algorithm in data and 'episode_rewards' in data[algorithm]:
            rewards_data = data[algorithm]['episode_rewards']
            
            # Extract mean and std series
            mean_rewards = rewards_data['mean_series']
            std_rewards = rewards_data['std_series']
            
            # Convert to numpy arrays for easier manipulation
            mean_rewards = np.array(mean_rewards)
            std_rewards = np.array(std_rewards)
            episodes = np.arange(1, len(mean_rewards) + 1)
            
            # Plot the mean learning curve
            ax.plot(episodes, mean_rewards, color=color, linewidth=2.5, 
                   label=f'{algorithm}', alpha=0.9)
            
            # Add confidence bands (mean ± std)
            ax.fill_between(episodes, 
                          mean_rewards - std_rewards, 
                          mean_rewards + std_rewards, 
                          color=color, alpha=0.2)
    
    # Customize the plot
    ax.set_xlabel('Training Episodes', fontsize=14, fontweight='bold')
    ax.set_ylabel('Episode Reward', fontsize=14, fontweight='bold')
    ax.set_title('Algorithm Learning Curves with 95% Confidence Bands', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Grid and styling
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=12, framealpha=0.9)
    
    # Set axis limits for better visualization
    max_episodes = max([len(data[alg]['episode_rewards']['mean_series']) 
                       for alg in colors.keys() if alg in data])
    ax.set_xlim(1, max_episodes)
    
    # Format y-axis to show rewards in thousands if values are large
    max_reward = max([max(data[alg]['episode_rewards']['mean_series']) 
                     for alg in colors.keys() if alg in data])
    if max_reward > 1000:
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x/1000:.1f}k'))
    
    # Improve layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Learning curves saved to: {output_path}")
    return fig, ax

def generate_performance_comparison(json_file_path, output_path="performance_comparison.png"):
    """
    Generate a bar chart comparing final performance across algorithms
    """
    
    # Load comprehensive data
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Extract final performance metrics
    algorithms = []
    mean_rewards = []
    success_rates = []
    
    # Define the order of algorithms for display
    algorithm_order = ['Random', 'DQN', 'Double DQN', 'Dueling DQN', 'PPO']
    
    for algorithm in algorithm_order:
        if algorithm in data:
            algorithms.append(algorithm)
            
            # Get asymptotic performance (final performance)
            if 'asymptotic_performance' in data[algorithm]:
                mean_rewards.append(data[algorithm]['asymptotic_performance']['mean'])
            elif 'mean_reward' in data[algorithm]:
                mean_rewards.append(data[algorithm]['mean_reward']['mean'])
            else:
                # Fallback: use last episode reward
                last_episode_reward = data[algorithm]['episode_rewards']['mean_series'][-1]
                mean_rewards.append(last_episode_reward)
            
            # Get success rate
            if 'success_rate' in data[algorithm]:
                success_rates.append(data[algorithm]['success_rate']['mean'] * 100)  # Convert to percentage
            else:
                success_rates.append(0)  # Default if not available
    
    # Create subplot with two y-axes
    fig, ax1 = plt.subplots(figsize=(12, 8))
    
    # Bar positions
    x_pos = np.arange(len(algorithms))
    
    # Plot mean rewards
    bars1 = ax1.bar(x_pos - 0.2, mean_rewards, 0.4, 
                    label='Mean Reward', color='#3498db', alpha=0.8)
    ax1.set_xlabel('Algorithms', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Mean Reward', fontsize=14, fontweight='bold', color='#3498db')
    ax1.tick_params(axis='y', labelcolor='#3498db')
    
    # Create second y-axis for success rates
    ax2 = ax1.twinx()
    bars2 = ax2.bar(x_pos + 0.2, success_rates, 0.4, 
                    label='Success Rate (%)', color='#e74c3c', alpha=0.8)
    ax2.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold', color='#e74c3c')
    ax2.tick_params(axis='y', labelcolor='#e74c3c')
    
    # Customize the plot
    ax1.set_title('Algorithm Performance Comparison', fontsize=16, fontweight='bold', pad=20)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(algorithms, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (reward, success) in enumerate(zip(mean_rewards, success_rates)):
        ax1.text(i - 0.2, reward + abs(reward) * 0.02, f'{reward:.0f}', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
        ax2.text(i + 0.2, success + 1, f'{success:.1f}%', 
                ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Add legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Performance comparison saved to: {output_path}")
    return fig, (ax1, ax2)

def main():
    """
    Main function to generate all visualizations
    """
    # Path to your comprehensive JSON file
    json_file_path = "experiments/comprehensive_evaluation_20250626_203337/comprehensive_results.json"
    
    # Check if file exists
    if not Path(json_file_path).exists():
        print(f"Error: File not found at {json_file_path}")
        print("Please check that the comprehensive results file exists at this location.")
        return
    
    try:
        # Load and inspect the data structure
        with open(json_file_path, 'r') as f:
            data = json.load(f)
        
        print("Available algorithms in data:")
        for algorithm in data.keys():
            print(f"  - {algorithm}")
        
        # Generate learning curves
        print("\nGenerating learning curves...")
        generate_learning_curves(json_file_path, "learning_curves.png")
        
        # Generate performance comparison
        print("Generating performance comparison...")
        generate_performance_comparison(json_file_path, "performance_comparison.png")
        
        print("\nAll visualizations generated successfully!")
        print("Files created:")
        print("- learning_curves.png")
        print("- performance_comparison.png")
        print(f"\nData source: {json_file_path}")
        
    except Exception as e:
        print(f"Error generating visualizations: {e}")
        print("Please check your JSON file format and try again.")

if __name__ == "__main__":
    main()