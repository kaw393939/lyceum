"""Streamlit application entry point."""
import sys
from pathlib import Path

# Add src directory to Python path to ensure proper imports
src_dir = (Path(__file__).resolve().parent / "src")
sys.path.insert(0, str(src_dir))

# Now import and run the main function
from rag_demo.main import main

if __name__ == "__main__":
    main()
