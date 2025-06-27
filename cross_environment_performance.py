import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def load_algorithm_results():
    """
    Load algorithm comparison results from all experiment directories
    """
    # Define the experiment directories and their corresponding grids
    experiment_data = [
        {
            'path': 'experiments/algorithm_comparison_20250626_111804/algorithm_summary.json',
            'grid': 'Assignment2 Main'
        },
        {
            'path': 'experiments/algorithm_comparison_20250626_114627/algorithm_summary.json',
            'grid': 'Corridor Test'
        },
        {
            'path': 'experiments/algorithm_comparison_20250626_115905/algorithm_summary.json',
            'grid': 'Maze Challenge'
        },
        {
            'path': 'experiments/algorithm_comparison_20250626_124142/algorithm_summary.json',
            'grid': 'Open Space'
        },
        {
            'path': 'experiments/algorithm_comparison_20250626_125532/algorithm_summary.json',
            'grid': 'Simple Restaurant'
        }
    ]
    
    # Load data from each experiment
    all_results = {}
    
    for exp_info in experiment_data:
        file_path = exp_info['path']
        grid_name = exp_info['grid']
        
        if Path(file_path).exists():
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                all_results[grid_name] = data
                print(f"✓ Loaded data for {grid_name}")
            except Exception as e:
                print(f"✗ Error loading {grid_name}: {e}")
        else:
            print(f"✗ File not found: {file_path}")
    
    return all_results

def create_focused_performance_visualization(all_results, output_path="algorithm_performance_comparison.png"):
    """
    Create a focused bar chart visualization showing algorithm performance across all environments
    """
    if not all_results:
        print("No data loaded. Cannot create visualization.")
        return None
    
    # Extract grid names and algorithms
    grid_names = list(all_results.keys())
    
    # Get all unique algorithms across all grids
    all_algorithms = set()
    for grid_data in all_results.values():
        all_algorithms.update(grid_data.keys())
    
    # Define algorithm order for consistent display
    algorithm_order = ['Random', 'DQN', 'Double DQN', 'Dueling DQN', 'PPO']
    algorithms = [alg for alg in algorithm_order if alg in all_algorithms]
    
    print(f"Found algorithms: {algorithms}")
    print(f"Found grids: {grid_names}")
    
    # Set up the figure with more space
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 14))
    
    # Colors for algorithms - professional color palette
    colors = {
        'Random': '#e74c3c',      # Red
        'DQN': '#3498db',         # Blue  
        'Double DQN': '#2ecc71',  # Green
        'Dueling DQN': '#f39c12', # Orange
        'PPO': '#9b59b6'          # Purple
    }
    
    # Prepare data for plotting
    x = np.arange(len(grid_names))
    width = 0.15  # Width of bars
    
    # Top subplot: Success Rates
    ax1.set_title('Algorithm Success Rates Across Assignment 2 Environments', 
                  fontsize=15, fontweight='bold', pad=15)
    
    for i, alg in enumerate(algorithms):
        success_rates = []
        
        for grid in grid_names:
            if alg in all_results[grid]:
                success_rates.append(all_results[grid][alg]['success_rate_avg'] * 100)
            else:
                success_rates.append(0)
        
        # Plot bars with offset
        bars = ax1.bar(x + i * width, success_rates, width, 
                      label=alg, color=colors[alg], alpha=0.8, 
                      edgecolor='white', linewidth=0.5)
        
        # Add value labels on top of bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                        f'{height:.1f}%', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')
    
    ax1.set_xlabel('Environment', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Success Rate (%)', fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax1.set_xticklabels(grid_names, fontsize=11)
    ax1.legend(loc='upper left', fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 110)
    
    # Bottom subplot: Final Evaluation Scores
    ax2.set_title('Algorithm Final Evaluation Scores Across Assignment 2 Environments', 
                  fontsize=15, fontweight='bold', pad=15)
    
    for i, alg in enumerate(algorithms):
        final_evals = []
        
        for grid in grid_names:
            if alg in all_results[grid]:
                final_evals.append(all_results[grid][alg]['final_eval_avg'])
            else:
                final_evals.append(0)
        
        # Plot bars with offset
        bars = ax2.bar(x + i * width, final_evals, width, 
                      label=alg, color=colors[alg], alpha=0.8,
                      edgecolor='white', linewidth=0.5)
        
        # Add value labels on top of bars
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if height > 0:
                ax2.text(bar.get_x() + bar.get_width()/2., height + abs(height)*0.02,
                        f'{height:.0f}', ha='center', va='bottom', 
                        fontsize=9, fontweight='bold')
    
    ax2.set_xlabel('Environment', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Final Evaluation Score', fontsize=14, fontweight='bold')
    ax2.set_xticks(x + width * (len(algorithms) - 1) / 2)
    ax2.set_xticklabels(grid_names, fontsize=11)
    ax2.legend(loc='lower right', fontsize=10, framealpha=0.9)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Adjust layout with reduced spacing for space conservation
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.25)
    
    # Save the plot
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    print(f"Performance comparison visualization saved to: {output_path}")
    return fig

def generate_summary_statistics(all_results):
    """
    Generate and print summary statistics across all environments
    """
    if not all_results:
        return
    
    print("\n" + "="*80)
    print("COMPREHENSIVE ALGORITHM PERFORMANCE SUMMARY")
    print("="*80)
    
    # Get all algorithms
    all_algorithms = set()
    for grid_data in all_results.values():
        all_algorithms.update(grid_data.keys())
    
    algorithm_order = ['Random', 'DQN', 'Double DQN', 'Dueling DQN', 'PPO']
    algorithms = [alg for alg in algorithm_order if alg in all_algorithms]
    
    # Calculate overall performance metrics
    overall_performance = {}
    
    for alg in algorithms:
        success_rates = []
        final_evals = []
        
        for grid_name, grid_data in all_results.items():
            if alg in grid_data:
                success_rates.append(grid_data[alg]['success_rate_avg'])
                final_evals.append(grid_data[alg]['final_eval_avg'])
        
        if success_rates and final_evals:
            overall_performance[alg] = {
                'avg_success_rate': np.mean(success_rates),
                'avg_final_eval': np.mean(final_evals),
                'success_std': np.std(success_rates),
                'eval_std': np.std(final_evals),
                'environments_tested': len(success_rates)
            }
    
    # Print overall ranking by success rate
    print(f"\n🏆 ALGORITHM RANKING BY SUCCESS RATE:")
    print("-" * 80)
    sorted_by_success = sorted(overall_performance.items(), 
                              key=lambda x: x[1]['avg_success_rate'], reverse=True)
    
    print(f"{'Rank':<4} {'Algorithm':<15} {'Avg Success Rate':<18} {'Std Dev':<10}")
    print("-" * 80)
    
    for rank, (alg_name, metrics) in enumerate(sorted_by_success, 1):
        print(f"{rank:<4} {alg_name:<15} "
              f"{metrics['avg_success_rate']:<18.1%} "
              f"±{metrics['success_std']:<9.1%}")
    
    # Print overall ranking by final evaluation
    print(f"\n🎯 ALGORITHM RANKING BY FINAL EVALUATION:")
    print("-" * 80)
    sorted_by_eval = sorted(overall_performance.items(), 
                           key=lambda x: x[1]['avg_final_eval'], reverse=True)
    
    print(f"{'Rank':<4} {'Algorithm':<15} {'Avg Final Eval':<15} {'Std Dev':<10}")
    print("-" * 80)
    
    for rank, (alg_name, metrics) in enumerate(sorted_by_eval, 1):
        print(f"{rank:<4} {alg_name:<15} "
              f"{metrics['avg_final_eval']:<15.1f} "
              f"±{metrics['eval_std']:<9.1f}")
    
    # Environment-specific analysis
    print(f"\n🌍 ENVIRONMENT-SPECIFIC PERFORMANCE:")
    print("-" * 80)
    
    for grid_name in all_results.keys():
        print(f"\n{grid_name}:")
        grid_data = all_results[grid_name]
        
        # Sort algorithms by performance in this environment
        env_performance = [(alg, metrics['final_eval_avg']) 
                          for alg, metrics in grid_data.items()]
        env_performance.sort(key=lambda x: x[1], reverse=True)
        
        for rank, (alg, score) in enumerate(env_performance, 1):
            success_rate = grid_data[alg]['success_rate_avg']
            print(f"  {rank}. {alg:<15}: {score:>8.1f} eval, {success_rate:>6.1%} success")
    
    # Overall insights
    if sorted_by_eval:
        champion = sorted_by_eval[0]
        print(f"\n🥇 OVERALL CHAMPION: {champion[0]}")
        print(f"   Average Success Rate: {champion[1]['avg_success_rate']:.1%}")
        print(f"   Average Final Evaluation: {champion[1]['avg_final_eval']:.1f}")
        print(f"   Tested on {champion[1]['environments_tested']} environments")
        
        # Calculate improvement over baseline
        if 'Random' in dict(sorted_by_eval):
            random_performance = dict(sorted_by_eval)['Random']['avg_final_eval']
            improvement = ((champion[1]['avg_final_eval'] - random_performance) / abs(random_performance)) * 100
            print(f"   Improvement over Random: {improvement:.1f}%")

def main():
    """
    Main function to generate focused cross-environment performance visualization
    """
    print("Loading algorithm comparison results from all Assignment 2 environments...")
    
    # Load all results
    all_results = load_algorithm_results()
    
    if not all_results:
        print("No data could be loaded. Please check file paths.")
        return
    
    # Generate focused visualization
    print("\nGenerating focused performance visualization...")
    create_focused_performance_visualization(all_results, "algorithm_performance_comparison.png")
    
    # Generate detailed summary statistics
    generate_summary_statistics(all_results)
    
    print(f"\n✅ Cross-environment analysis complete!")
    print(f"📊 Focused visualization saved as: algorithm_performance_comparison.png")
    print(f"💡 The visualization shows both success rates and evaluation scores")
    print(f"   for easy comparison of algorithm performance across all environments.")

if __name__ == "__main__":
    main()