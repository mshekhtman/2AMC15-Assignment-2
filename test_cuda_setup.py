#!/usr/bin/env python3
"""
Simple CUDA Test for Assignment 2 DQN Training
Quick verification that CUDA works with your DQN agent.
"""
import torch
import sys
import time
from pathlib import Path

def print_header(title):
    print(f"\n{'='*50}")
    print(f"{title:^50}")
    print(f"{'='*50}")

def test_cuda_basic():
    """Test basic CUDA setup."""
    print_header("CUDA BASIC TEST")
    
    print(f"Python: {sys.version.split()[0]}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available!")
        return False
    
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU Count: {torch.cuda.device_count()}")
    
    # Basic GPU info
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  Memory: {props.total_memory / 1024**3:.1f} GB")
        print(f"  Compute: {props.major}.{props.minor}")
    
    return True

def test_gpu_computation():
    """Test GPU computation works."""
    print_header("GPU COMPUTATION TEST")
    
    if not torch.cuda.is_available():
        print("❌ No GPU for testing")
        return False
    
    try:
        # Simple computation test
        device = torch.device("cuda")
        print(f"Using device: {device}")
        
        # Create tensors and compute
        x = torch.randn(500, 500, device=device)
        y = torch.randn(500, 500, device=device)
        
        start_time = time.time()
        z = torch.matmul(x, y)
        torch.cuda.synchronize()
        gpu_time = time.time() - start_time
        
        print(f"✅ GPU computation successful!")
        print(f"Matrix multiplication time: {gpu_time:.4f}s")
        print(f"Result shape: {z.shape}")
        
        # Memory usage
        memory_used = torch.cuda.memory_allocated() / 1024**2
        print(f"GPU memory used: {memory_used:.1f} MB")
        
        return True
        
    except Exception as e:
        print(f"❌ GPU computation failed: {e}")
        return False

def test_dqn_agent():
    """Test DQN agent GPU compatibility."""
    print_header("DQN AGENT TEST")
    
    try:
        # Add path for imports
        sys.path.insert(0, str(Path.cwd()))
        
        # Import and create DQN agent
        from agents.DQN_agent import DQNAgent
        
        print("Creating DQN agent...")
        agent = DQNAgent(state_dim=8, action_dim=4, verbose=False)
        
        print(f"✅ DQN agent created!")
        print(f"Agent device: {agent.device}")
        print(f"Network device: {next(agent.q_net.parameters()).device}")
        
        # Test forward pass
        test_state = torch.randn(1, 8).to(agent.device)
        with torch.no_grad():
            q_values = agent.q_net(test_state)
        
        print(f"✅ Forward pass successful!")
        print(f"Q-values: {q_values.cpu().numpy().flatten()}")
        
        # Test action selection
        action = agent.take_action(test_state.cpu().numpy().flatten())
        print(f"✅ Action selection: {action} ({agent.get_action_name(action)})")
        
        return True
        
    except Exception as e:
        print(f"❌ DQN test failed: {e}")
        return False

def quick_performance_test():
    """Quick performance comparison."""
    print_header("PERFORMANCE TEST")
    
    if not torch.cuda.is_available():
        print("❌ No GPU for performance test")
        return
    
    try:
        size = 1000
        iterations = 3
        
        # CPU test
        print("Testing CPU...")
        cpu_times = []
        for i in range(iterations):
            x = torch.randn(size, size)
            y = torch.randn(size, size)
            start = time.time()
            z = torch.matmul(x, y)
            cpu_times.append(time.time() - start)
        
        avg_cpu = sum(cpu_times) / len(cpu_times)
        
        # GPU test
        print("Testing GPU...")
        device = torch.device("cuda")
        gpu_times = []
        
        # Warmup
        x = torch.randn(100, 100, device=device)
        y = torch.randn(100, 100, device=device)
        torch.matmul(x, y)
        torch.cuda.synchronize()
        
        for i in range(iterations):
            x = torch.randn(size, size, device=device)
            y = torch.randn(size, size, device=device)
            torch.cuda.synchronize()
            start = time.time()
            z = torch.matmul(x, y)
            torch.cuda.synchronize()
            gpu_times.append(time.time() - start)
        
        avg_gpu = sum(gpu_times) / len(gpu_times)
        speedup = avg_cpu / avg_gpu if avg_gpu > 0 else 0
        
        print(f"📊 Results:")
        print(f"  CPU time: {avg_cpu:.4f}s")
        print(f"  GPU time: {avg_gpu:.4f}s")
        print(f"  Speedup: {speedup:.1f}x")
        
        if speedup > 5:
            print("🚀 Excellent GPU acceleration!")
        elif speedup > 2:
            print("👍 Good GPU acceleration")
        else:
            print("⚠️  Limited speedup")
            
    except Exception as e:
        print(f"❌ Performance test failed: {e}")

def main():
    """Run all tests."""
    print("🎮 CUDA Setup Test for Assignment 2")
    
    # Run tests
    cuda_ok = test_cuda_basic()
    
    if cuda_ok:
        gpu_ok = test_gpu_computation()
        dqn_ok = test_dqn_agent()
        quick_performance_test()
        
        print_header("SUMMARY")
        
        print(f"CUDA Available: {'✅' if cuda_ok else '❌'}")
        print(f"GPU Computation: {'✅' if gpu_ok else '❌'}")
        print(f"DQN Compatibility: {'✅' if dqn_ok else '❌'}")
        
        if cuda_ok and gpu_ok and dqn_ok:
            print("\n🎉 SUCCESS! Ready for GPU-accelerated DQN training!")
            print("\nTry running:")
            print("python train.py grid_configs/A1_grid.npy --agent_type dqn --episodes 20 --no_gui")
        else:
            print("\n⚠️  Some tests failed. Check the errors above.")
    else:
        print("\n❌ CUDA not available. Check your installation.")

if __name__ == "__main__":
    main()