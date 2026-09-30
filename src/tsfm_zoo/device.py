import subprocess
from pathlib import Path

import torch


def pick_device(arg: str = "auto") -> str:
    """`arg`, or with "auto" the least-used GPU that ~/.claude/free-gpu reports free, else "cpu"."""
    if arg != "auto":
        return arg
    tool = Path.home() / ".claude" / "free-gpu"
    if torch.cuda.is_available() and tool.exists():
        free = subprocess.run([str(tool)], capture_output=True, text=True).stdout.strip().split(",")[0]
        if free and free != "-1":
            return f"cuda:{free}"
    return "cpu"
