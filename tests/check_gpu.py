import torch

print(f"PyTorch version : {torch.__version__}")
print(f"CUDA available  : {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version    : {torch.version.cuda}")
    print(f"Device name     : {torch.cuda.get_device_name(0)}")
    print(f"Device count    : {torch.cuda.device_count()}")
else:
    print(
        "\nCUDA not available to PyTorch. This usually means the CPU-only "
        "PyTorch wheel is installed rather than a CUDA-enabled build, even "
        "if you have a working NVIDIA GPU (e.g. for Ollama). Check `nvidia-smi` "
        "to confirm the driver sees the GPU, then reinstall PyTorch with the "
        "correct CUDA build for your driver version from pytorch.org."
    )