import numpy as np

def handle_non_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()  # Convert NumPy arrays to lists
    # Handle other non-serializable elements here
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

def ndarray_to_list(arr):
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    elif isinstance(arr, list):
        return [ndarray_to_list(item) for item in arr]
    else:
        return arr