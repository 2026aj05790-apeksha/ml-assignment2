"""Compatibility launcher for the production training module."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).parent / "src" / "train.py"), run_name="__main__")
