import csv
import os

import pycuda.autoinit  # noqa: F401 — side-effect: creates CUDA context
import pycuda.driver as cuda

cuda.init()

device = cuda.Device(0)
name = device.name()
num_sms = device.get_attribute(cuda.device_attribute.MULTIPROCESSOR_COUNT)
clock_rate_ghz = device.get_attribute(cuda.device_attribute.CLOCK_RATE) / 1e6
compute_capability = device.compute_capability()

# Per-arch defaults. Ratios are expressed relative to FP32 CUDA-core FMA
# throughput (the "base" FP32 TFLOPS computed below). Sources: NVIDIA
# arch whitepapers and TechPowerUp/GPU-Z flagship specs.

if compute_capability >= (9, 0):  # Hopper (H100, H200)
    cores_per_sm = 128
    tensor_cores_per_sm = 4
    fp64_ratio = 1 / 2
    fp16_ratio = 16          # dense tensor FP16
    fp64_tensor_ratio = 1
    tf_32_ratio = 8
    bf_16_ratio = 16

elif compute_capability >= (8, 0):  # Ampere / Ada Lovelace
    if "A100" in name:
        cores_per_sm = 64
        tensor_cores_per_sm = 4
        fp64_ratio = 1 / 2
        fp16_ratio = 16
        fp64_tensor_ratio = 1
        tf_32_ratio = 8
        bf_16_ratio = 16
    else:  # RTX 30xx / 40xx / Ax000 workstation
        cores_per_sm = 128
        tensor_cores_per_sm = 4
        fp64_ratio = 1 / 64
        fp16_ratio = 1        # CUDA cores do FP16 at FP32 rate on consumer Ampere/Ada
        fp64_tensor_ratio = 0
        tf_32_ratio = 1       # tensor TF32 ≈ FP32 TFLOPS (dense)
        bf_16_ratio = 2       # tensor BF16 ≈ 2× FP32 (dense)

elif compute_capability == (7, 5):  # Turing (RTX 20xx, GTX 16xx)
    cores_per_sm = 64
    tensor_cores_per_sm = 0 if "GTX 16" in name else 8
    fp64_ratio = 1 / 32
    fp16_ratio = 2
    fp64_tensor_ratio = 0
    tf_32_ratio = 0
    bf_16_ratio = 0

elif compute_capability == (7, 0):  # Volta (Tesla V100)
    cores_per_sm = 64
    tensor_cores_per_sm = 8
    fp64_ratio = 1 / 2
    fp16_ratio = 2
    fp64_tensor_ratio = 0
    tf_32_ratio = 0
    bf_16_ratio = 0

else:  # Pascal and older
    cores_per_sm = 64
    tensor_cores_per_sm = 0
    fp64_ratio = 1 / 32
    fp16_ratio = 0
    fp64_tensor_ratio = 0
    tf_32_ratio = 0
    bf_16_ratio = 0

total_cores = num_sms * cores_per_sm
num_tensor_cores = num_sms * tensor_cores_per_sm

fp32_flops = total_cores * 2 * clock_rate_ghz / 1000  # TFLOPS (FMA = 2 FLOP/cycle)
fp64_flops = fp32_flops * fp64_ratio
fp16_flops = fp32_flops * fp16_ratio
fp64_tensor_flops = fp64_flops * fp64_tensor_ratio
tf32_flops = fp32_flops * tf_32_ratio
bf16_flops = fp32_flops * bf_16_ratio

config = {
    'GPU Name': name,
    'Compute Capability (cuda version)': f'{compute_capability}',
    'Number of SMs': num_sms,
    'Cores per SM': cores_per_sm,
    'Total CUDA Cores': total_cores,
    'Number of Tensor Cores': num_tensor_cores,
    'Clock Rate (GHz) boost': clock_rate_ghz,
    'FP32 FLOPS (TFLOPS)': fp32_flops,
    'TF32 FLOPS (TFLOPS)': tf32_flops,
    'FP16 FLOPS (TFLOPS)': fp16_flops,
    'BF16 FLOPS (TFLOPS)': bf16_flops,
    'FP64 FLOPS (GFLOPS)': fp64_flops * 1000,
    'FP64 TENSOR FLOPS (TFLOPS)': fp64_tensor_flops,
}

output_dir = os.environ.get('DATA_DIR', '/app/data')
os.makedirs(output_dir, exist_ok=True)
with open(os.path.join(output_dir, 'gpu_config.csv'), 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(config.keys())
    writer.writerow(config.values())
