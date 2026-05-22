import torch
import torch.nn as nn
import time
import numpy as np
import argparse
import matplotlib.pyplot as plt
from mamba import SSM

# Argument parser
parser = argparse.ArgumentParser(description='Test SSM latency with different parameter combinations')
parser.add_argument('--d_model', type=str, default='8,16,32,64,128,256,512,1024,2048,4096,8192,16384,32768,65536', help='Model dimensions as comma-separated values')
parser.add_argument('--d_state', type=str, default='4,8,16', help='State dimensions as comma-separated values')
parser.add_argument('--ssm_ratio', type=str, default='2,4', help='SSM ratios as comma-separated values')
parser.add_argument('--seq_len', type=int, default=256, help='Sequence length (default: 256)')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size (default: 1)')
args = parser.parse_args()

# Parse multiple values
d_model_values = [int(x.strip()) for x in args.d_model.split(',')]
d_state_values = [int(x.strip()) for x in args.d_state.split(',')]
ssm_ratio_values = [int(x.strip()) for x in args.ssm_ratio.split(',')]

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("_" * 120)
print("COMMAND LINE ARGUMENTS")
print("_" * 120)
print(f"d_model values:    {d_model_values}")
print(f"d_state values:    {d_state_values}")
print(f"ssm_ratio values:  {ssm_ratio_values}")
print(f"seq_len:           {args.seq_len}")
print(f"batch_size:        {args.batch_size}")
print(f"device:            {device}")
print("_" * 120 + "\n")

total_configs = len(d_model_values) * len(d_state_values) * len(ssm_ratio_values)
print(f"Testing {len(d_model_values)} × {len(d_state_values)} × {len(ssm_ratio_values)} = {total_configs} configurations...\n")

# Fixed test parameters
warmup_runs = 5
test_runs = 10

results = []

for d_state in d_state_values:
    for ssm_ratio in ssm_ratio_values:
        for d_model in d_model_values:
            try:
                print(f"Testing d_model={d_model}, d_state={d_state}, ssm_ratio={ssm_ratio}...", end=" ")
                
                # Create SSM module
                ssm = SSM(
                    d_model=d_model,
                    d_state=d_state,
                    ssm_ratio=ssm_ratio
                ).to(device)
                
                # Create input tensor (batch_size from args)
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
                
                results.append({
                    'd_model': d_model,
                    'd_state': d_state,
                    'ssm_ratio': ssm_ratio,
                    'num_params': num_params,
                    'avg_latency': avg_latency,
                    'std_latency': std_latency,
                    'min_latency': min_latency,
                    'max_latency': max_latency,
                    'memory_mb': memory_allocated,
                })
                
                print(f"✓ {avg_latency:.4f}ms")
                
                # Clear memory
                del ssm
                del x
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                    
            except RuntimeError as e:
                print(f"✗ Error: {str(e)[:60]}")
                continue

print("\n" + "=" * 120)
print("LATENCY COMPARISON RESULTS")
print("=" * 120)
print(f"{'d_model':<10} {'d_state':<10} {'ssm_ratio':<12} {'Parameters':<15} {'Avg Latency (ms)':<18} {'Throughput (tok/s)':<20}")
print("-" * 120)

for result in results:
    print(f"{result['d_model']:<10} {result['d_state']:<10} {result['ssm_ratio']:<12} {result['num_params']:<15,} {result['avg_latency']:<18.4f} {result['throughput']:<20,.0f}")

print("=" * 120)

# Create visualizations
if results:
    print("\nGenerating plots...\n")
    
    # Plot 1: Latency vs d_model for each d_state and ssm_ratio combination
    fig, axes = plt.subplots(len(d_state_values), len(ssm_ratio_values), figsize=(14, 10))
    if len(d_state_values) == 1 or len(ssm_ratio_values) == 1:
        axes = np.atleast_2d(axes)
    
    for idx_state, d_state in enumerate(d_state_values):
        for idx_ratio, ssm_ratio in enumerate(ssm_ratio_values):
            ax = axes[idx_state, idx_ratio] if axes.ndim > 1 else axes
            
            subset = [r for r in results if r['d_state'] == d_state and r['ssm_ratio'] == ssm_ratio]
            if subset:
                d_models = [r['d_model'] for r in subset]
                latencies = [r['avg_latency'] for r in subset]
                
                ax.plot(d_models, latencies, marker='o', linewidth=2, markersize=8, color='blue')
                ax.set_xlabel('d_model', fontsize=10)
                ax.set_ylabel('Latency (ms)', fontsize=10)
                ax.set_title(f'd_state={d_state}, ssm_ratio={ssm_ratio}', fontsize=11, fontweight='bold')
                ax.grid(True, alpha=0.3)
                if max(d_models) > 1000:
                    ax.set_xscale('log')
    
    plt.tight_layout()
    plt.savefig('latency_vs_dmodel.png', dpi=100, bbox_inches='tight')
    print("✓ Saved: latency_vs_dmodel.png")
    
    # Plot 2: Latency vs d_state for each d_model and ssm_ratio combination
    num_plots = len(d_model_values) * len(ssm_ratio_values)
    grid_size = int(np.ceil(np.sqrt(num_plots)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(14, 12))
    axes = axes.flatten()
    
    plot_idx = 0
    for d_model in d_model_values:
        for ssm_ratio in ssm_ratio_values:
            ax = axes[plot_idx]
            
            subset = [r for r in results if r['d_model'] == d_model and r['ssm_ratio'] == ssm_ratio]
            if subset:
                d_states = [r['d_state'] for r in subset]
                latencies = [r['avg_latency'] for r in subset]
                
                ax.bar(range(len(d_states)), latencies, color='green', alpha=0.7)
                ax.set_xticks(range(len(d_states)))
                ax.set_xticklabels(d_states)
                ax.set_xlabel('d_state', fontsize=9)
                ax.set_ylabel('Latency (ms)', fontsize=9)
                ax.set_title(f'd_model={d_model}, ssm_ratio={ssm_ratio}', fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
            plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('latency_vs_dstate.png', dpi=100, bbox_inches='tight')
    print("✓ Saved: latency_vs_dstate.png")
    
    # Plot 3: Latency vs ssm_ratio for each d_model and d_state combination
    num_plots = len(d_model_values) * len(d_state_values)
    grid_size = int(np.ceil(np.sqrt(num_plots)))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(14, 12))
    axes = axes.flatten()
    
    plot_idx = 0
    for d_model in d_model_values:
        for d_state in d_state_values:
            ax = axes[plot_idx]
            
            subset = [r for r in results if r['d_model'] == d_model and r['d_state'] == d_state]
            if subset:
                ratios = [r['ssm_ratio'] for r in subset]
                latencies = [r['avg_latency'] for r in subset]
                
                ax.bar(range(len(ratios)), latencies, color='orange', alpha=0.7)
                ax.set_xticks(range(len(ratios)))
                ax.set_xticklabels(ratios)
                ax.set_xlabel('ssm_ratio', fontsize=9)
                ax.set_ylabel('Latency (ms)', fontsize=9)
                ax.set_title(f'd_model={d_model}, d_state={d_state}', fontsize=10, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
            plot_idx += 1
    
    # Hide unused subplots
    for i in range(plot_idx, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig('latency_vs_ssm_ratio.png', dpi=100, bbox_inches='tight')
    print("✓ Saved: latency_vs_ssm_ratio.png")
    
    # Plot 4: Heatmap - d_model vs d_state (for each ssm_ratio)
    for ssm_ratio in ssm_ratio_values:
        subset = [r for r in results if r['ssm_ratio'] == ssm_ratio]
        
        # Create matrix
        latency_matrix = []
        for d_state in sorted(set(r['d_state'] for r in subset)):
            row = []
            for d_model in sorted(set(r['d_model'] for r in subset)):
                match = [r for r in subset if r['d_state'] == d_state and r['d_model'] == d_model]
                if match:
                    row.append(match[0]['avg_latency'])
                else:
                    row.append(np.nan)
            latency_matrix.append(row)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(latency_matrix, cmap='YlOrRd', aspect='auto')
        
        d_models_unique = sorted(set(r['d_model'] for r in subset))
        d_states_unique = sorted(set(r['d_state'] for r in subset))
        
        ax.set_xticks(range(len(d_models_unique)))
        ax.set_yticks(range(len(d_states_unique)))
        ax.set_xticklabels(d_models_unique)
        ax.set_yticklabels(d_states_unique)
        ax.set_xlabel('d_model', fontsize=12, fontweight='bold')
        ax.set_ylabel('d_state', fontsize=12, fontweight='bold')
        ax.set_title(f'Latency Heatmap (ssm_ratio={ssm_ratio})', fontsize=13, fontweight='bold')
        
        # Add text annotations
        for i in range(len(d_states_unique)):
            for j in range(len(d_models_unique)):
                if not np.isnan(latency_matrix[i][j]):
                    text = ax.text(j, i, f'{latency_matrix[i][j]:.2f}',
                                  ha="center", va="center", color="black", fontsize=9)
        
        plt.colorbar(im, ax=ax, label='Latency (ms)')
        plt.tight_layout()
        plt.savefig(f'latency_heatmap_ssm_ratio_{ssm_ratio}.png', dpi=100, bbox_inches='tight')
        print(f"✓ Saved: latency_heatmap_ssm_ratio_{ssm_ratio}.png")

print("\n" + "=" * 120)
print("All plots saved!")
print("=" * 120)



