import torch

def get_device():
    """
    Check for CUDA, HPU, or CPU availability and return the appropriate device.
    Priority order: CUDA > HPU > CPU

    Returns:
        torch.device: The available device (cuda, hpu, or cpu)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        try:
            import habana_frameworks.torch.hpu as hthpu

            if hthpu.is_available():
                return torch.device("hpu")
        except ImportError:
            pass

    return torch.device("cpu")


def print_device_info(device):
    """
    Print information about the selected device.

    Args:
        device (torch.device): The device to get information about
    """
    if device.type == "cuda":
        print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"CUDA version: {torch.version.cuda}")
    elif device.type == "hpu":
        import habana_frameworks.torch.hpu as hthpu

        print("Using HPU device")
    else:
        print("Using CPU device")


def safe_model_load(model_class, model_name, **kwargs):
    """
    Safely load a model with fallback options for reshape errors.

    Args:
        model_class: The model class to instantiate
        model_name: The model name or path
        **kwargs: Additional arguments for model loading

    Returns:
        The loaded model and the device it was loaded on
    """
    device = get_device()

    # Try loading with original settings
    try:
        if device.type == "cuda":
            model = model_class.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="auto", **kwargs
            )
        else:
            model = model_class.from_pretrained(
                model_name, torch_dtype=torch.float32, device_map=device, **kwargs
            ).to(device)
        return model, device
    except Exception as e:
        print(f"Failed to load model with optimal settings: {e}")
        print("Falling back to CPU with basic settings...")

        # Fallback to CPU
        try:
            # Remove GPU-specific arguments
            safe_kwargs = {
                k: v
                for k, v in kwargs.items()
                if k not in ["device_map", "quantization_config"]
            }

            model = model_class.from_pretrained(
                model_name, torch_dtype=torch.float32, **safe_kwargs
            ).to(torch.device("cpu"))
            return model, torch.device("cpu")
        except Exception as e2:
            raise RuntimeError(
                f"Failed to load model even with fallback settings: {e2}"
            )


def move_to_device(data, device):
    """
    Safely move data to a device, handling various data types.

    Args:
        data: Data to move (tensor, dict, list, etc.)
        device: Target device

    Returns:
        Data moved to the specified device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    elif isinstance(data, tuple):
        return tuple(move_to_device(item, device) for item in data)
    else:
        return data
