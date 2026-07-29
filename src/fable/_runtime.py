"""Fixed runtime profile owned by the installed executable."""

from typing import Literal

import torch

FIT_BATCH_SIZE = 64
EVALUATION_BATCH_SIZE = 512
FIT_PRECISION: Literal["bf16-mixed"] = "bf16-mixed"

FLOAT32_MATMUL_PRECISION = "high"
CUDA_MATMUL_ALLOW_TF32 = True
CUDNN_ALLOW_TF32 = True


def configure_torch() -> None:
    torch.set_float32_matmul_precision(FLOAT32_MATMUL_PRECISION)
    torch.backends.cuda.matmul.allow_tf32 = CUDA_MATMUL_ALLOW_TF32
    torch.backends.cudnn.allow_tf32 = CUDNN_ALLOW_TF32
