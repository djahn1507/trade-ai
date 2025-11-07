
"""Compatibility wrapper that prefers the real NumPy package when available."""
from __future__ import annotations

import importlib.util
import os
import sys
from types import ModuleType
from typing import List, Tuple


def _remove_repo_paths(repo_root: str) -> List[Tuple[int, str]]:
    removed: List[Tuple[int, str]] = []
    normalized_root = os.path.abspath(repo_root)
    for index in range(len(sys.path) - 1, -1, -1):
        entry = sys.path[index]
        if os.path.abspath(entry) == normalized_root:
            removed.append((index, entry))
            sys.path.pop(index)
    removed.reverse()
    return removed


def _restore_repo_paths(removed: List[Tuple[int, str]]) -> None:
    for index, value in removed:
        sys.path.insert(index, value)


def _load_real_numpy() -> ModuleType | None:
    repo_root = os.path.dirname(os.path.dirname(__file__))
    removed_paths = _remove_repo_paths(repo_root)
    placeholder = sys.modules.pop(__name__, None)
    loaded_module: ModuleType | None = None
    try:
        spec = importlib.util.find_spec('numpy')
        if spec is None or spec.origin is None:
            return None
        if os.path.abspath(spec.origin) == os.path.abspath(__file__):
            return None
        module = importlib.util.module_from_spec(spec)
        loader = spec.loader
        if loader is None:  # pragma: no cover - defensive, mirrors importlib
            return None
        loader.exec_module(module)
        loaded_module = module
        return module
    finally:
        if loaded_module is None and placeholder is not None:
            sys.modules[__name__] = placeholder
        _restore_repo_paths(removed_paths)


_real_numpy = _load_real_numpy()
if _real_numpy is not None:
    sys.modules[__name__] = _real_numpy
    globals().update(_real_numpy.__dict__)
else:  # pragma: no cover - exercised in environments without real numpy
    from ._stub import *  # type: ignore[F403]
