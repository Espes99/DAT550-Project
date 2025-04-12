import os
from pathlib import Path

def prepare_unique_output_path(path: str) -> str:
    """
    Ensures the given output path is unique by incrementing the folder directly
    after a protected prefix if necessary. Creates the final path.

    Returns:
        str: A safe, unique path.
    """
    PROTECTED_PREFIXES = [
        ("logs", "training"), ("logs", "testing"),
        ("outputs", "training"), ("outputs", "testing"),
        ("analysis", "training"), ("analysis", "testing"),
        ("analysis", "comparison")
    ]

    path = Path(path)
    parts = path.parts

    # Identify the protected prefix
    prefix_len = 0
    for protected in PROTECTED_PREFIXES:
        for i in range(len(parts) - len(protected)):
            if parts[i:i+len(protected)] == protected:
                prefix_len = i + len(protected)
                break

    if prefix_len == 0:
        raise ValueError(f"Path '{path}' does not match any protected prefix")

    base_name = parts[prefix_len]
    suffix = Path(*parts[prefix_len + 1:])
    parent = Path(*parts[:prefix_len])

    # Increment if collision
    candidate = parent / base_name
    counter = 1
    while (candidate / suffix).exists():
        candidate = parent / f"{base_name}{counter}"
        counter += 1

    full_path = candidate / suffix
    os.makedirs(full_path, exist_ok=True)
    return str(full_path)