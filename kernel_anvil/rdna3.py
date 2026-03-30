"""RDNA3 hardware constants and optimization heuristics."""
from dataclasses import dataclass


@dataclass(frozen=True)
class GpuSpec:
    name: str
    gfx: str
    peak_bandwidth_gbs: float
    cu_count: int
    simds_per_cu: int
    max_waves_per_simd: int
    vgpr_per_simd: int
    vgpr_granule: int
    lds_per_cu_bytes: int
    lds_granule_bytes: int
    wave_size: int

    def max_vgpr_waves(self, vgpr_count: int) -> int:
        """Max concurrent waves per SIMD given VGPR usage."""
        if vgpr_count == 0:
            return self.max_waves_per_simd
        alloc = ((vgpr_count + self.vgpr_granule - 1) // self.vgpr_granule) * self.vgpr_granule
        return min(self.max_waves_per_simd, self.vgpr_per_simd // alloc)

    def max_lds_waves(self, lds_bytes: int, threads_per_wg: int) -> int:
        """Max concurrent waves per SIMD given LDS usage."""
        if lds_bytes == 0:
            return self.max_waves_per_simd
        wgs_per_cu = self.lds_per_cu_bytes // max(lds_bytes, self.lds_granule_bytes)
        waves_per_wg = (threads_per_wg + self.wave_size - 1) // self.wave_size
        total_waves = wgs_per_cu * waves_per_wg
        return min(self.max_waves_per_simd, total_waves // self.simds_per_cu)

    def occupancy(self, vgpr_count: int, lds_bytes: int, threads_per_wg: int) -> tuple[float, str]:
        """Returns (occupancy_pct, limiting_factor)."""
        vgpr_w = self.max_vgpr_waves(vgpr_count)
        lds_w = self.max_lds_waves(lds_bytes, threads_per_wg)
        active = min(vgpr_w, lds_w)
        pct = active / self.max_waves_per_simd * 100
        if vgpr_w < lds_w:
            factor = "vgpr"
        elif lds_w < vgpr_w:
            factor = "lds"
        else:
            factor = "balanced"
        return pct, factor


# Known RDNA3 GPUs
GFX1100 = GpuSpec(
    name="Radeon RX 7900 XTX", gfx="gfx1100",
    peak_bandwidth_gbs=960, cu_count=96, simds_per_cu=2,
    max_waves_per_simd=10, vgpr_per_simd=1536, vgpr_granule=8,
    lds_per_cu_bytes=98304, lds_granule_bytes=512, wave_size=32,
)

GFX1101 = GpuSpec(
    name="Radeon RX 7900 XT", gfx="gfx1101",
    peak_bandwidth_gbs=800, cu_count=84, simds_per_cu=2,
    max_waves_per_simd=10, vgpr_per_simd=1536, vgpr_granule=8,
    lds_per_cu_bytes=98304, lds_granule_bytes=512, wave_size=32,
)

GFX1102 = GpuSpec(
    name="Radeon RX 7800 XT", gfx="gfx1102",
    peak_bandwidth_gbs=576, cu_count=60, simds_per_cu=2,
    max_waves_per_simd=10, vgpr_per_simd=1536, vgpr_granule=8,
    lds_per_cu_bytes=98304, lds_granule_bytes=512, wave_size=32,
)

GPU_SPECS = {"gfx1100": GFX1100, "gfx1101": GFX1101, "gfx1102": GFX1102}


def detect_gpu() -> GpuSpec | None:
    """Detect current AMD GPU and return its spec."""
    import subprocess
    try:
        out = subprocess.run(["rocm-smi", "--showproductname"], capture_output=True, text=True, timeout=5)
        for gfx, spec in GPU_SPECS.items():
            if gfx in out.stdout.lower() or spec.name.lower() in out.stdout.lower():
                return spec
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    # Fallback: try torch
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0).lower()
            if "7900 xtx" in name or "gfx1100" in name:
                return GFX1100
            if "7900 xt" in name and "xtx" not in name:
                return GFX1101
            if "7800" in name:
                return GFX1102
    except Exception:
        pass
    return None


# Proven optimization heuristics from AEGIS kernel tournaments
HEURISTICS = [
    "Scalar inner loops (tl.static_range) beat 2D tile patterns on RDNA3",
    "LUT-based dequantization beats arithmetic dequant (ALU dependency chains)",
    "Two-pass SPLIT_K beats atomic_add for SPLIT_K > 4 (no FP16 precision loss)",
    "Atomic SPLIT_K is faster for small shapes (less launch overhead)",
    "Row-major access pattern wins over tl.trans() (kills coalesced advantage)",
    "Fuse same-input kernels: gate+up sharing input x saves 61% vs separate calls",
    "VGPR allocation granularity is 8 registers -- plan register usage accordingly",
    "Coalesced loads: adjacent threads in a wave should access sequential bytes",
    "Register spilling to scratch memory (VRAM) destroys performance",
    "BLOCK_N=32-64 maximizes occupancy; BLOCK_N=128-256 maximizes parallelism per wave",
]
