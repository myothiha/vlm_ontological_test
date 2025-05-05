# setup_path.py

import os
import sys

def add_project_root():
    current_dir = os.path.dirname(__file__)
    project_root = os.path.abspath(current_dir)
    if project_root not in sys.path:
        sys.path.append(project_root)
