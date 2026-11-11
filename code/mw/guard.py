"""
mw.guard — memory safety.

A tool that hangs the user's laptop is worse than no tool. Loading a model with
default settings duplicates the state dict in RAM during load, and on a machine
with zram swap (compressed RAM) memory pressure causes a spiral: zram itself
consumes RAM, so the kernel never recovers gracefully.

This module refuses to start a load that is unlikely to fit, and picks a dtype
that does fit. It is deliberately conservative.
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path


def _linux_meminfo(key: str) -> float:
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith(key):
                    return int(line.split()[1]) / 1024 / 1024
    except OSError:
        pass
    return 0.0


def _psutil_mem() -> tuple[float, float] | None:
    """(available_gb, total_gb) via psutil, or None if psutil is unusable."""
    try:
        import psutil  # cross-platform, transitive dep of accelerate
    except ImportError:
        return None
    try:
        vm = psutil.virtual_memory()
        return vm.available / 1e9, vm.total / 1e9
    except Exception:  # noqa: BLE001 - psutil can raise on exotic platforms
        return None


def available_gb() -> float:
    """Available RAM, platform-agnostic.

    psutil works on Linux, macOS and Windows; the /proc fallback exists so the
    package still functions if psutil is absent on Linux. Returning 0.0 on an
    unknown platform would make require() refuse every load, so we return a
    sentinel of -1.0 and treat it as "unknown, do not block".
    """
    mem = _psutil_mem()
    if mem is not None:
        return mem[0]
    if platform.system() == "Linux":
        return _linux_meminfo("MemAvailable:")
    return -1.0  # unknown: do not block


def total_gb() -> float:
    mem = _psutil_mem()
    if mem is not None:
        return mem[1]
    if platform.system() == "Linux":
        return _linux_meminfo("MemTotal:")
    return -1.0


def zram_swap() -> bool:
    """zram swap is RAM-backed: pressure spirals instead of degrading.

    Linux-only. macOS compresses memory in the kernel (different failure mode)
    and Windows uses a pagefile, so this returns False there by design. We
    DETECT zram as a risk factor; we never depend on it existing.
    """
    if platform.system() != "Linux":
        return False
    try:
        with open("/proc/swaps", encoding="utf-8", errors="ignore") as f:
            return any("zram" in line for line in f)
    except OSError:
        return False


def model_file_gb(path: str) -> float | None:
    """Best-effort size of the weights on disk."""
    p = Path(path)
    cand = p / "model.safetensors"
    if cand.exists():
        return cand.stat().st_size / 1e9
    if p.is_file() and p.suffix == ".safetensors":
        return p.stat().st_size / 1e9
    total = 0.0
    if p.is_dir():
        for f in p.glob("*.safetensors"):
            total += f.stat().st_size
    return (total / 1e9) if total else None


def _header(path: str) -> dict | None:
    import json
    import struct

    p = Path(path)
    f = p / "model.safetensors" if p.is_dir() else p
    try:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            return json.loads(fh.read(n))
    except (OSError, ValueError):
        return None


def _tensor_stats(path: str) -> tuple[int, int, dict] | None:
    """(num_params, num_bytes, params_by_dtype). Reads the header only."""
    h = _header(path)
    if not h:
        return None
    n_params = 0
    n_bytes = 0
    by_dtype: dict[str, int] = {}
    for k, v in h.items():
        if k == "__metadata__" or not isinstance(v, dict):
            continue
        n = 1
        for d in v.get("shape", []):
            n *= int(d)
        off = v.get("data_offsets", [0, 0])
        n_params += n
        n_bytes += int(off[1]) - int(off[0])
        by_dtype[v.get("dtype")] = by_dtype.get(v.get("dtype"), 0) + n
    return n_params, n_bytes, by_dtype


def source_dtype(path: str) -> str | None:
    """Dominant dtype weighted by PARAMETER COUNT, not the first tensor.

    The first tensor is often a 1-D norm stored in fp32, which made a
    mostly-fp16 model look like fp32 and under-estimated memory.
    """
    st = _tensor_stats(path)
    if not st or not st[2]:
        return None
    return max(st[2], key=st[2].get)


def bytes_per_param(path: str) -> float:
    """Average bytes per parameter, from the header (handles mixed precision)."""
    st = _tensor_stats(path)
    if not st or st[0] == 0:
        return 2.0  # typical LLM checkpoint
    return st[1] / st[0]


def unsafe_downcast(path: str, dtype: str | None) -> bool:
    """True if the requested dtype would overflow the source dtype."""
    if dtype != "float16":
        return False
    return source_dtype(path) == "BF16"


def plan(path: str, dtype: str | None = None) -> dict:
    """Decide dtype and whether the load is safe. Returns a plan, never loads."""
    avail = available_gb()
    disk_gb = model_file_gb(path)

    # D9: a repo id has no local files, so the guard used to wave it through.
    # Resolve its size from the Hub instead; if that fails, refuse rather than
    # gamble with the user's machine.
    if disk_gb is None and not Path(path).exists():
        try:
            from huggingface_hub import model_info
            info = model_info(path, files_metadata=True)
            total = sum((s.size or 0) for s in info.siblings
                        if s.rfilename.endswith(".safetensors"))
            disk_gb = (total / 1e9) if total else None
        except Exception:  # noqa: BLE001 - offline/404/rate-limit all mean unknown
            disk_gb = None
    if disk_gb is None:
        if os.environ.get("MOD_WEI_ALLOW_OOM") == "1":
            return {"ok": True, "dtype": dtype or "float16",
                    "reason": "size unknown (override)", "available_gb": avail}
        return {"ok": False, "dtype": dtype or "float16",
                "reason": "size unknown: cannot verify the load fits",
                "available_gb": avail}

    # Derive parameter count from the header, not from file size (G1 D8).
    # A 5 GB fp16 file is ~2.5B params, and fp32 loading needs ~10 GB of weights,
    # not 5 GB. The old formula dropped the x2 for files >= 8 GB.
    bpp = bytes_per_param(path)
    params_gb = disk_gb / max(bpp, 0.5)
    need_fp32 = params_gb * 4.0 * 1.4              # + ~40% activations/transient
    need_fp16 = params_gb * 2.0 * 1.4

    if dtype is None:
        dtype = "float32" if avail > need_fp32 + 4 else "float16"
    need = need_fp32 if dtype == "float32" else need_fp16

    # avail < 0 means "unknown platform": do not block, just warn.
    ok = True if avail < 0 else avail > need + 2.0
    return {
        "ok": ok,
        "dtype": dtype,
        "disk_gb": round(disk_gb, 2),
        "need_gb": round(need, 2),
        "available_gb": round(avail, 2),
        "zram": zram_swap(),
    }


def require(path: str, dtype: str | None = None) -> str:
    """Abort with actionable advice instead of hanging the machine."""
    p = plan(path, dtype)
    if p["ok"]:
        return p["dtype"]
    sys.stderr.write(
        "\nREFUSING TO LOAD — this would likely freeze the machine.\n"
        f"  model on disk : {p.get('disk_gb')} GB\n"
        f"  needs approx  : {p.get('need_gb')} GB ({p['dtype']})\n"
        f"  available     : {p.get('available_gb')} GB"
        + ("  (zram swap detected: memory pressure will not degrade gracefully)\n"
           if p.get("zram") else "\n")
        + "\nFree memory first:\n"
        "  ollama ps && ollama stop <model>     # unload served models\n"
        "  close VMs / browser tabs\n"
        "or force a smaller footprint:\n"
        "  adapter.load(dtype=torch.float16)\n"
        "  MOD_WEI_ALLOW_OOM=1 to override (not recommended)\n\n"
    )
    if os.environ.get("MOD_WEI_ALLOW_OOM") == "1":
        sys.stderr.write("MOD_WEI_ALLOW_OOM=1 set — continuing anyway.\n")
        return p["dtype"]
    raise MemoryError("insufficient memory; load refused")
