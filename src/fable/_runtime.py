"""Fixed runtime profile owned by the installed executable."""

import os

import torch

FIT_BATCH_SIZE = 64
EVALUATION_BATCH_SIZE = 64

NUM_WORKERS = 4
PIN_MEMORY = True
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = True

DETERMINISTIC = True
BENCHMARK = False
FLOAT32_MATMUL_PRECISION = "high"
CUDA_MATMUL_ALLOW_TF32 = True
CUDNN_ALLOW_TF32 = True


def configure_torch() -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(DETERMINISTIC)
    torch.backends.cudnn.deterministic = DETERMINISTIC
    torch.backends.cudnn.benchmark = BENCHMARK
    torch.set_float32_matmul_precision(FLOAT32_MATMUL_PRECISION)
    torch.backends.cuda.matmul.allow_tf32 = CUDA_MATMUL_ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = CUDNN_ALLOW_TF32
