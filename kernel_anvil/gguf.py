"""GGUF model parser -- extract tensor shapes and quant types.

Reads a GGUF file and identifies the unique kernel workloads (quant_type, N, K)
so kernel-anvil knows which shapes to optimize.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TensorInfo:
    name: str
    shape: tuple[int, ...]  # (rows, cols) for 2D weight tensors
    quant_type: str  # "Q4_K", "Q8_0", "F16", etc.
    size_bytes: int


@dataclass
class ModelProfile:
    name: str  # model name from metadata
    architecture: str  # "llama", "qwen2", "qwen3", etc.
    tensors: list[TensorInfo]  # all tensors
    unique_shapes: dict[tuple[str, int, int], int] = field(default_factory=dict)
    # (quant_type, N, K) -> count -- only 2D weight tensors


def _get_string_field(reader, key: str) -> str | None:
    """Extract a string metadata field from a GGUFReader."""
    if key not in reader.fields:
        return None
    f = reader.fields[key]
    if len(f.data) > 0:
        part = f.parts[f.data[0]]
        if hasattr(part, "dtype"):
            return bytes(part).decode("utf-8", errors="replace")
    return None


def parse_gguf(path: str | Path) -> ModelProfile:
    """Parse a GGUF file and extract tensor metadata.

    Uses the ``gguf`` Python package (from llama.cpp) for reading.

    Args:
        path: Path to a .gguf file.

    Returns:
        ModelProfile with all tensors and unique 2D workload shapes.

    Raises:
        FileNotFoundError: If path does not exist.
        ImportError: If the gguf package is not installed.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"GGUF file not found: {path}")

    from gguf import GGUFReader

    reader = GGUFReader(str(path))

    name = _get_string_field(reader, "general.name") or path.stem
    architecture = _get_string_field(reader, "general.architecture") or "unknown"

    tensors: list[TensorInfo] = []
    shape_counter: Counter[tuple[str, int, int]] = Counter()

    for t in reader.tensors:
        shape = tuple(int(x) for x in t.shape)
        quant_type = t.tensor_type.name
        info = TensorInfo(
            name=t.name,
            shape=shape,
            quant_type=quant_type,
            size_bytes=int(t.n_bytes),
        )
        tensors.append(info)

        # Only 2D tensors are GEMV workloads
        if len(shape) == 2:
            n, k = shape
            shape_counter[(quant_type, n, k)] += 1

    return ModelProfile(
        name=name,
        architecture=architecture,
        tensors=tensors,
        unique_shapes=dict(shape_counter),
    )


def print_model_summary(profile: ModelProfile) -> None:
    """Print a summary of the model's kernel workloads."""
    total_2d = sum(1 for t in profile.tensors if len(t.shape) == 2)
    total_1d = sum(1 for t in profile.tensors if len(t.shape) != 2)
    total_bytes = sum(t.size_bytes for t in profile.tensors)

    print(f"Model: {profile.name}")
    print(f"Architecture: {profile.architecture}")
    print(f"Tensors: {len(profile.tensors)} ({total_2d} weight, {total_1d} other)")
    print(f"Total size: {total_bytes / 1e9:.2f} GB")
    print()
    print(f"Unique GEMV workloads ({len(profile.unique_shapes)}):")
    print(f"  {'Quant':<8} {'N':>8} {'K':>8} {'Count':>6}")
    print(f"  {'-----':<8} {'-----':>8} {'-----':>8} {'-----':>6}")
    for (quant, n, k), count in sorted(profile.unique_shapes.items()):
        print(f"  {quant:<8} {n:>8} {k:>8} {count:>6}")
