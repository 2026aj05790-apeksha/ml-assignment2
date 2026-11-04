"""Compatibility launcher for the production Streamlit application."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).parent / "src" / "app.py"), run_name="__main__")
