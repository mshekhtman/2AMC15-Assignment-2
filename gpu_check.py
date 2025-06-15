# gpu_check.py - Run this to check your current GPU setup
import torch
import sys

print("=== GPU ACCELERATION CHECK ===")
print(f"Python version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# Check CUDA availability
print(f"\nCUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    # Test GPU computation
    print(f"\nCurrent device: {torch.cuda.current_device()}")
    
    # Simple GPU test
    try:
        x = torch.randn(1000, 1000).cuda()
        y = torch.randn(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print("✅ GPU computation test: SUCCESS")
        print(f"Result tensor device: {z.device}")
    except Exception as e:
        print(f"❌ GPU computation test: FAILED - {e}")
        
else:
    print("❌ CUDA not available - using CPU only")
    print("\nTo enable GPU acceleration:")
    print("1. Install NVIDIA GPU drivers")
    print("2. Install CUDA Toolkit")
    print("3. Install PyTorch with CUDA support")

# Check which device DQN will use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"\nDQN will use device: {device}")

if torch.cuda.is_available():
    memory_allocated = torch.cuda.memory_allocated() / 1024**3
    memory_reserved = torch.cuda.memory_reserved() / 1024**3
    print(f"GPU Memory - Allocated: {memory_allocated:.2f} GB, Reserved: {memory_reserved:.2f} GB")