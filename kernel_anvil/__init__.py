"""kernel-anvil - Profile-guided Triton kernel optimizer for AMD/RDNA3."""
__version__ = "0.1.0"

from kernel_anvil.model import SmithyLinear, optimize  # noqa: F401
from kernel_anvil.codegen import generate_config_header  # noqa: F401
