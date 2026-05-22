import torch
import torch.nn as nn
import time
import numpy as np
import argparse
from mamba import SSM

# Argument parser
parser = argparse.ArgumentParser(description='Test SSM latency with different d_model values')
parser.add_argument('--d_model', type=str, default='8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536,131072,262144,524288,1048576,2097152,4194304,8388608', help='Model dimensions as comma-separated values (default: 64,96,128,192,256)')
parser.add_argument('--d_state', type=int, default=4, help='State dimension (default: 4)')
parser.add_argument('--ssm_ratio', type=int, default=2, help='SSM ratio (default: 2)')
parser.add_argument('--seq_len', type=int, default=256, help='Sequence length (default: 256)')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size (default: 4)')
args = parser.parse_args()

# Parse d_model values
d_model_values = [int(x.strip()) for x in args.d_model.split(',')]

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Testing d_model values: {d_model_values}")
print(f"Fixed parameters: d_state={args.d_state}, ssm_ratio={args.ssm_ratio}")
print(f"Fixed input: seq_len={args.seq_len}, batch_size={args.batch_size}\n")

# Fixed test parameters
warmup_runs = 5
test_runs = 10

results = []

for d_model in d_model_values:
    try:
        print(f"Testing d_model={d_model}...")
        
        # Create SSM module
        ssm = SSM(
            d_model=d_model,
            d_state=args.d_state,
            ssm_ratio=args.ssm_ratio
        ).to(device)
        
        # Create input tensor
        x = torch.randn(args.batch_size, args.seq_len, d_model).to(device)
        
        # Count parameters
        num_params = sum(p.numel() for p in ssm.parameters())
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                _ = ssm(x)
        
        # Synchronize GPU if using CUDA
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Measure latency
        latencies = []
        with torch.no_grad():
            for i in range(test_runs):
                start_time = time.perf_counter()
                _ = ssm(x)
                if device.type == 'cuda':
                    torch.cuda.synchronize()
                end_time = time.perf_counter()
                latencies.append((end_time - start_time) * 1000)  # Convert to ms
        
        avg_latency = np.mean(latencies)
        std_latency = np.std(latencies)
        min_latency = np.min(latencies)
        max_latency = np.max(latencies)
        
        # Memory usage
        if device.type == 'cuda':
            memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # Convert to MB
        else:
            memory_allocated = 0
        
        # Throughput calculation (tokens per second)
        tokens_per_sec = (args.batch_size * args.seq_len) / (avg_latency / 1000)
        
        results.append({
            'd_model': d_model,
            'num_params': num_params,
            'avg_latency': avg_latency,
            'std_latency': std_latency,
            'min_latency': min_latency,
            'max_latency': max_latency,
            'memory_mb': memory_allocated,
            'throughput': tokens_per_sec
        })
        
        # Clear memory
        del ssm
        del x
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    except RuntimeError as e:
        print(f"Error with d_model={d_model}: {str(e)[:100]}")
        continue

print("\n" + "=" * 110)
print("LATENCY COMPARISON RESULTS")
print("=" * 110)
print(f"{'d_model':<10} {'Parameters':<15} {'Avg Latency':<15} {'Std Dev':<12} {'Memory (MB)':<12} {'Throughput':<15}")
print("-" * 110)

for result in results:
    print(f"{result['d_model']:<10} {result['num_params']:<15,} {result['avg_latency']:<15.4f} {result['std_latency']:<12.4f} {result['memory_mb']:<12.2f} {result['throughput']:<15,.0f}")

print("=" * 110)

# Analysis: How latency changes when d_model is doubled
print("\n" + "=" * 110)
print("LATENCY SCALING ANALYSIS (How latency changes when d_model is doubled)")
print("=" * 110)

if len(results) >= 2:
    for i in range(len(results) - 1):
        current = results[i]
        next_result = results[i + 1]
        
        d_model_ratio = next_result['d_model'] / current['d_model']
        latency_ratio = next_result['avg_latency'] / current['avg_latency']
        param_ratio = next_result['num_params'] / current['num_params']
        
        print(f"\nd_model: {current['d_model']} → {next_result['d_model']} (ratio: {d_model_ratio:.2f}x)")
        print(f"  Latency increase: {latency_ratio:.2f}x ({current['avg_latency']:.4f}ms → {next_result['avg_latency']:.4f}ms)")
        print(f"  Parameters increase: {param_ratio:.2f}x ({current['num_params']:,} → {next_result['num_params']:,})")
        print(f"  Memory increase: {next_result['memory_mb'] / current['memory_mb']:.2f}x ({current['memory_mb']:.2f}MB → {next_result['memory_mb']:.2f}MB)")

print("\n" + "=" * 110)



