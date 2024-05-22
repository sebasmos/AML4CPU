import torch
def bytes_to_mb(memory_bytes):
    """
    Takes the memory usage in bytes as input and returns the memory usage converted to megabytes (MB).
    """
    return memory_bytes / (1024 * 1024)
def model_memory_usage_alternative(model, input_dtype=torch.float32):
    total_params = sum(p.numel() for p in model.parameters())
    dtype_size = torch.tensor([], dtype=input_dtype).element_size()  # Size of the data type
    total_memory = total_params * dtype_size  # Total memory in bytes
    return total_memory / (1024 ** 2)