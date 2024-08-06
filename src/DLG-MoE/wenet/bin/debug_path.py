import os
import sys


def setup_pythonpath():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_parent_dir = os.path.dirname(os.path.dirname(current_dir))
    if parent_parent_dir not in sys.path:
        sys.path.append(parent_parent_dir)
    os.environ['PYTHONPATH'] = parent_parent_dir + os.pathsep + os.environ.get('PYTHONPATH', '')


setup_pythonpath()
