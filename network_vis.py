import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_network_architecture_data(file_path):
    """
    Load network architecture comparison data
    """
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"✓ Successfully loaded network architecture data")
        return data
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None

def create_architecture_comparison_visualization(data, output_path="network_architecture_comparison.png"):
    """
    Create a focused visualization comparing network architecture performance
    """
    if not data:
        print("No data available for visualization.")
        return None
    
    # Extract algorithm types and network architectures
    algorithms = list(data.keys())  # ['dqn', 'ppo']
    
    # Get all network architectures (should be consistent across algorithms)
    architectures = list(data[algorithms[0]].keys())
    print(f"Found algorithms: {algorithms}")
    print(f"Found architectures: {architectures}")
    
    # Set up the figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Define colors for algorithms
    colors = {
        'dqn': '#3498db',     # Blue
        'ppo': '#e74c3c'      # Red
    }
    
    # Define algorithm labels for display
    algorithm_labels = {
        'dqn': 'DQN',
        'ppo': 'PPO'
    }
    
    # Prepare data for plotting
    x = np.arange(len(architectures))
    width = 0.35  # Width of bars
    
    # Extract success rates for each algorithm and architecture
    success_rates = {}
    for alg in algorithms:
        success_rates[alg] = []
        for arch in architectures:
            if arch in data[alg]:
                # Convert success rate to percentage
                rate = data[alg][arch]['success_rate'] * 100
                success_rates[alg].append(rate)
            else:
                success_rates[alg].append(0)
    
    # Create grouped bar chart
    bars = {}
    for i, alg in enumerate(algorithms):
        offset = (i - 0.5) * width
        bars[alg] = ax.bar(x + offset, success_rates[alg], width, 
                          label=algorithm_labels[alg], color=colors[alg], 
                          alpha=0.8, edgecolor='white', linewidth=1)
        
        # Add value labels on top of bars
        for j, bar in enumerate(bars[alg]):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}%', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
    
    # Customize the plot
    ax.set_xlabel('Network Architecture', fontsize=14, fontweight='bold')
    ax.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    ax.set_title('Network Architecture Impact on Algorithm Success Rates', 
                fontsize=16, fontweight='bold', pad=20)
    
    # Format architecture names for better display
    arch_labels = []
    for arch in architectures:
        # Convert from "Tiny_Net" to "Tiny Net" format
        formatted = arch.replace('_', ' ')
        arch_labels.append(formatted)
    
    ax.set_xticks(x)
    ax.set_xticklabels(arch_labels, fontsize=12)
    ax.legend(loc='upper left', fontsize=12, framealpha=0.9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max([max(success_rates[alg]) for alg in algorithms]) + 10)
    
    # Add some insights as text
    # Find best performing architecture for each algorithm
    best_arch = {}
    for alg in algorithms:
        best_idx = np.argmax(success_rates[alg])
        best_arch[alg] = {
            'name': architectures[best_idx],
            'rate': success_rates[alg][best_idx]
        }
    
    # Add text box with key insights
    insight_text = f"Best Architectures:\n"
    for alg in algorithms:
        arch_name = best_arch[alg]['name'].replace('_', ' ')
        insight_text += f"• {algorithm_labels[alg]}: {arch_name} ({best_arch[alg]['rate']:.1f}%)\n"
    
    # Remove the last newline
    insight_text = insight_text.rstrip('\n')
    
    # Add text box in the upper right
    ax.text(0.98, 0.98, insight_text, transform=ax.transAxes, 
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8),
           verticalalignment='top', horizontalalignment='right',
           fontsize=10, fontweight='bold')
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Network architecture comparison saved to: {output_path}")
    return fig

def generate_architecture_analysis(data):
    """
    Generate detailed analysis of network architecture performance
    """
    if not data:
        return
    
    print("\n" + "="*70)
    print("NETWORK ARCHITECTURE PERFORMANCE ANALYSIS")
    print("="*70)
    
    algorithms = list(data.keys())
    architectures = list(data[algorithms[0]].keys())
    
    # Analyze each algorithm
    for alg in algorithms:
        print(f"\n🔧 {alg.upper()} Algorithm Analysis:")
        print("-" * 50)
        
        # Sort architectures by success rate for this algorithm
        arch_performance = []
        for arch in architectures:
            if arch in data[alg]:
                success_rate = data[alg][arch]['success_rate']
                mean_reward = data[alg][arch]['mean_reward']
                std_reward = data[alg][arch]['std_reward']
                final_10_avg = data[alg][arch]['final_10_avg']
                
                arch_performance.append({
                    'name': arch,
                    'success_rate': success_rate,
                    'mean_reward': mean_reward,
                    'std_reward': std_reward,
                    'final_10_avg': final_10_avg
                })
        
        # Sort by success rate
        arch_performance.sort(key=lambda x: x['success_rate'], reverse=True)
        
        print(f"{'Rank':<4} {'Architecture':<12} {'Success Rate':<12} {'Mean Reward':<12} {'Final 10 Avg':<12}")
        print("-" * 70)
        
        for rank, arch_data in enumerate(arch_performance, 1):
            arch_name = arch_data['name'].replace('_', ' ')
            print(f"{rank:<4} {arch_name:<12} "
                  f"{arch_data['success_rate']:<12.1%} "
                  f"{arch_data['mean_reward']:<12.1f} "
                  f"{arch_data['final_10_avg']:<12.1f}")
    
    # Cross-algorithm comparison
    print(f"\n🏆 CROSS-ALGORITHM ARCHITECTURE COMPARISON:")
    print("-" * 70)
    
    print(f"{'Architecture':<12} {'DQN Success':<12} {'PPO Success':<12} {'Best Algorithm':<15}")
    print("-" * 70)
    
    for arch in architectures:
        arch_name = arch.replace('_', ' ')
        dqn_success = data['dqn'][arch]['success_rate'] * 100
        ppo_success = data['ppo'][arch]['success_rate'] * 100
        
        if dqn_success > ppo_success:
            best_alg = f"DQN (+{dqn_success - ppo_success:.1f}%)"
        elif ppo_success > dqn_success:
            best_alg = f"PPO (+{ppo_success - dqn_success:.1f}%)"
        else:
            best_alg = "Tie"
        
        print(f"{arch_name:<12} {dqn_success:<12.1f}% {ppo_success:<12.1f}% {best_alg:<15}")
    
    # Overall insights
    print(f"\n💡 KEY INSIGHTS:")
    print("-" * 30)
    
    # Find overall best architecture
    overall_best = None
    best_avg_success = 0
    
    for arch in architectures:
        avg_success = (data['dqn'][arch]['success_rate'] + data['ppo'][arch]['success_rate']) / 2
        if avg_success > best_avg_success:
            best_avg_success = avg_success
            overall_best = arch
    
    if overall_best:
        print(f"• Best Overall Architecture: {overall_best.replace('_', ' ')} "
              f"(Avg: {best_avg_success:.1%})")
    
    # Architecture complexity analysis
    complexity_order = ['Tiny_Net', 'Small_Net', 'Medium_Net', 'Large_Net', 'Deep_Net']
    print(f"• Architecture Complexity vs Performance:")
    
    for arch in complexity_order:
        if arch in architectures:
            dqn_rate = data['dqn'][arch]['success_rate']
            ppo_rate = data['ppo'][arch]['success_rate']
            avg_rate = (dqn_rate + ppo_rate) / 2
            print(f"  - {arch.replace('_', ' ')}: {avg_rate:.1%} average success")

def main():
    """
    Main function to generate network architecture comparison visualization
    """
    # Path to the network architecture results
    file_path = "experiments/simplified_ablation_20250626_163903/network_architecture_results.json"
    
    # Check if file exists
    if not Path(file_path).exists():
        print(f"Error: File not found at {file_path}")
        print("Please check that the network architecture results file exists at this location.")
        return
    
    print("Loading network architecture comparison data...")
    
    # Load the data
    data = load_network_architecture_data(file_path)
    
    if not data:
        print("Failed to load data. Cannot create visualization.")
        return
    
    # Generate visualization
    print("Generating network architecture comparison visualization...")
    create_architecture_comparison_visualization(data, "network_architecture_comparison.png")
    
    # Generate detailed analysis
    generate_architecture_analysis(data)
    
    print(f"\n✅ Network architecture analysis complete!")
    print(f"📊 Visualization saved as: network_architecture_comparison.png")
    print(f"🔍 The visualization shows success rate impact of different network architectures")
    print(f"   for both DQN and PPO algorithms, helping identify optimal complexity levels.")

if __name__ == "__main__":
    main()