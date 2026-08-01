"""Fixed runtime profile owned by the installed executable."""

import torch

FIT_BATCH_SIZE = 64
EVALUATION_BATCH_SIZE = 512


def configure_torch() -> None:
    torch.set_float32_matmul_precision("high")
    torch.backends.cudnn.allow_tf32 = True
