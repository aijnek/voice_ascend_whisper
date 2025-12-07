"""Device detection and management for Apple Silicon MPS backend."""

import torch


def get_device():
    """
    Detect and configure optimal device for training/inference.

    Returns:
        torch.device: MPS device if available, otherwise CPU
    """
    if torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("Warning: MPS not built in PyTorch, falling back to CPU")
            return torch.device("cpu")

        print("Using MPS (Metal Performance Shaders) device")
        return torch.device("mps")
    else:
        print("MPS not available, using CPU")
        return torch.device("cpu")


def clear_mps_cache():
    """Clear MPS cache to free memory."""
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


def print_device_info():
    """Print information about available devices."""
    print("=" * 50)
    print("Device Information")
    print("=" * 50)
    print(f"PyTorch version: {torch.__version__}")
    print(f"MPS available: {torch.backends.mps.is_available()}")
    print(f"MPS built: {torch.backends.mps.is_built()}")

    device = get_device()
    print(f"Selected device: {device}")
    print("=" * 50)

    return device
