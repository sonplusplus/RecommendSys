# check_gpu.py

import os
import sys




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
    print("NO GPU DETECTED")
    print("="*70)