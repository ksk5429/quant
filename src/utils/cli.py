"""Shared CLI utilities for finding and invoking the Claude Code CLI."""

from __future__ import annotations

import os
import shutil


def find_claude_binary() -> str:
    """Find the claude CLI binary cross-platform.

    Search order:
    1. PATH (shutil.which)
    2. ~/.local/bin/claude[.exe]
    3. ~/.claude/bin/claude[.exe]
    4. Fallback to bare "claude" (will fail with clear error)
    """
    found = shutil.which("claude")
    if found:
        return found
    home = os.path.expanduser("~")
    for name in ["claude.exe", "claude"]:
        for subdir in [".local/bin", ".claude/bin"]:
            candidate = os.path.join(home, subdir, name)
            if os.path.isfile(candidate):
                return candidate
    return "claude"
