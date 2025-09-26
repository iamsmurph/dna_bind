"""Hashing and lightweight caching helpers for prefilter indexing.

The helpers here are intentionally minimal and avoid importing heavy deps.
They support:
 - computing per-directory signatures from file size and mtime_ns
 - building a stable global hash across many directories
 - atomic JSON writes and simple filesystem locks (NFS-friendly)
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Dict, Optional, Tuple


def _stat_record(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {"path": None, "size": 0, "mtime_ns": 0, "realpath": None}
    try:
        st = os.stat(path)
        return {
            "path": path,
            "size": int(getattr(st, "st_size", 0)),
            "mtime_ns": int(getattr(st, "st_mtime_ns", int(getattr(st, "st_mtime", 0) * 1e9))),
            "realpath": os.path.realpath(path),
        }
    except Exception:
        return {"path": path, "size": 0, "mtime_ns": 0, "realpath": None}


def compute_dir_signature(files: Dict[str, Optional[str]]) -> Dict[str, object]:
    """Build a deterministic signature dict for a prediction directory.

    files maps logical keys (e.g., "pt", "pae", "pde", "cif", "contact") to paths.
    """
    sig: Dict[str, object] = {}
    for key in sorted(files.keys()):
        sig[key] = _stat_record(files[key])
    return sig


def stable_global_hash(dir_to_sig: Dict[str, Dict[str, object]]) -> str:
    """Compute a stable SHA1 over the sorted mapping of dir -> signature."""
    # Build a compact structure to hash
    items = []
    for d in sorted(dir_to_sig.keys()):
        sig = dir_to_sig[d]
        # Reduce fields to primitives for stability
        reduced = {k: {"size": v.get("size", 0), "mtime_ns": v.get("mtime_ns", 0), "realpath": v.get("realpath")}
                   for k, v in sorted(sig.items())}
        items.append((d, reduced))
    payload = json.dumps(items, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha1(payload).hexdigest()


def atomic_write_json(target_path: str, obj: object) -> None:
    """Atomically write JSON to target_path via a .tmp and os.replace."""
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    tmp_path = target_path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(obj, f)
    os.replace(tmp_path, target_path)


def read_json(path: str) -> Optional[object]:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return None


def try_acquire_lock(lock_dir: str) -> bool:
    """Try to create a directory lock. Returns True if acquired."""
    try:
        os.mkdir(lock_dir)
        return True
    except Exception:
        return False


def release_lock(lock_dir: str) -> None:
    try:
        os.rmdir(lock_dir)
    except Exception:
        pass


def wait_for_file(path: str, timeout_s: float = 600.0, poll_s: float = 0.5) -> bool:
    """Wait for a file to exist up to timeout. Returns True if present."""
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if os.path.exists(path):
            return True
        time.sleep(poll_s)
    return os.path.exists(path)

