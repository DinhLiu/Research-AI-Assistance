#!/usr/bin/env python3
"""Double-click launcher. Prefer the project's installed virtual environment."""
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
python = ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
if python.exists() and Path(sys.prefix).resolve() != (ROOT / '.venv').resolve():
    os.execv(str(python), [str(python), str(Path(__file__).resolve()), *sys.argv[1:]])
os.chdir(ROOT)
from research_assistant.ui.server import main

if __name__ == '__main__':
    main()
