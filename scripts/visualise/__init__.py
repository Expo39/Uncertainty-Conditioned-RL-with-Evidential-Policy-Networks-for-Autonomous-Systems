"""
@file __init__.py
@brief Visualisation package for 2D bird's-eye training and replay views.

Sets the matplotlib backend before any submodule imports pyplot,
preventing the "headless" backend from locking in.
"""

import os

import matplotlib
matplotlib.use("TkAgg" if os.environ.get("DISPLAY") else "Agg")
