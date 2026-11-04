"""Compatibility launcher for the production README generator."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).parent / "src" / "build_readme.py"), run_name="__main__")
