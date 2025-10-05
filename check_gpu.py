import os
import sys

# Force CUDA paths
cuda_path = "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2"
try:
    os.add_dll_directory(f"{cuda_path}/bin")
    os.add_dll_directory(f"{cuda_path}/lib/x64")
    print("✓ Added CUDA DLL directories")
except:
    print("⚠ Could not add CUDA directories")

import tensorflow as tf

print("\n" + "="*70)
print("TENSORFLOW GPU CHECK")
print("="*70)

print(f"\nTensorFlow version: {tf.__version__}")
print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

gpus = tf.config.list_physical_devices('GPU')
print(f"\nGPU devices found: {len(gpus)}")

if gpus:
    for i, gpu in enumerate(gpus):
        print(f"  GPU {i}: {gpu.name}")
        details = tf.config.experimental.get_device_details(gpu)
        print(f"    Compute capability: {details.get('compute_capability', 'N/A')}")
    
    print("\n" + "="*70)
    print("✓✓✓ SUCCESS! Testing GPU computation...")
    print("="*70)
    
    with tf.device('/GPU:0'):
        a = tf.random.normal([1000, 1000])
        b = tf.random.normal([1000, 1000])
        c = tf.matmul(a, b)
    
    print("✓ Matrix multiplication on GPU: PASSED")
    print("✓✓✓ GPU is working correctly! ✓✓✓\n")
    
else:
    print("\n" + "="*70)
    print("✗✗✗ NO GPU DETECTED ✗✗✗")
    print("="*70)
    
    from tensorflow.python.platform import build_info
    print("\nBuild info:")
    for key, value in build_info.build_info.items():
        if 'cuda' in key.lower() or 'gpu' in key.lower():
            print(f"  {key}: {value}")
    
    print("\nTroubleshooting:")
    print("  1. Check if TensorFlow is GPU build (is_cuda_build should be True)")
    print("  2. Try: pip install tensorflow==2.17.0")
    print("  3. Or install Miniconda and use conda install tensorflow-gpu")
    print("  4. Restart computer after changing PATH\n")