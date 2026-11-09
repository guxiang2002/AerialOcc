"""Evaluate exported AerialOcc predictions without importing torch or MMDet."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from aerialocc.evaluation.eval26 import main

if __name__ == '__main__':
    main()
