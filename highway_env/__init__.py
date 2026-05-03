"""Compatibility package for editable installs.

This module exposes the project as ``highway_env`` while keeping the original
source tree under ``FlowIS``.
"""

from __future__ import annotations

from importlib import import_module
from pathlib import Path

# Make submodule lookups like ``highway_env.envs`` resolve to ``FlowIS/envs``.
__path__ = [str(Path(__file__).resolve().parent.parent / "FlowIS")]
__version__ = "1.10.1"


def register_highway_envs():
    """Compatibility hook used by Gymnasium plugin entry points."""
    flowis = import_module("FlowIS")
    return flowis._register_highway_envs()
