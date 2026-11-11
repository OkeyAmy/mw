"""Guard tests — cross-platform memory safety. No model, no network."""
import platform

import pytest
from mw import guard


def test_available_gb_returns_a_number():
    a = guard.available_gb()
    assert isinstance(a, float)
    # -1.0 means "unknown platform"; anything else must be non-negative
    assert a == -1.0 or a >= 0.0


def test_available_is_plausible_on_known_platforms():
    if platform.system() in ("Linux", "Darwin", "Windows"):
        a = guard.available_gb()
        # either the sentinel (psutil missing + non-Linux) or a real number
        assert a == -1.0 or a > 0.01


def test_zram_detection_is_linux_only():
    if platform.system() != "Linux":
        assert guard.zram_swap() is False
    else:
        assert isinstance(guard.zram_swap(), bool)


def test_plan_does_not_block_when_memory_unknown(tmp_path):
    """The cross-platform bug: returning 0.0 for unknown RAM would refuse every
    load on macOS/Windows. Unknown must mean 'do not block'."""
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"\0" * 1024)
    real = guard.available_gb
    try:
        guard.available_gb = lambda: -1.0
        p = guard.plan(str(tmp_path))
        assert p["ok"] is True
    finally:
        guard.available_gb = real


def test_plan_blocks_when_memory_is_clearly_insufficient(tmp_path):
    f = tmp_path / "model.safetensors"
    f.write_bytes(b"\0" * 1024)
    # pretend a 20 GB model on a 1 GB machine
    real_avail, real_size = guard.available_gb, guard.model_file_gb
    try:
        guard.available_gb = lambda: 1.0
        guard.model_file_gb = lambda p: 20.0
        p = guard.plan(str(tmp_path), dtype="float16")
        assert p["ok"] is False
        with pytest.raises(MemoryError):
            guard.require(str(tmp_path), "float16")
    finally:
        guard.available_gb, guard.model_file_gb = real_avail, real_size


def test_dtype_auto_selects_fp16_when_tight(tmp_path):
    real_avail, real_size = guard.available_gb, guard.model_file_gb
    try:
        guard.available_gb = lambda: 6.0
        guard.model_file_gb = lambda p: 5.0
        p = guard.plan(str(tmp_path))
        assert p["dtype"] == "float16"
    finally:
        guard.available_gb, guard.model_file_gb = real_avail, real_size
