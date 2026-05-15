"""
======================================================
core/state.py
------------------------------------------------------
Responsibility:
- Store runtime state related to training execution

Design rules:
- In-memory only (DO NOT persist to disk)
- Single source of truth for active training process
- No business logic — only state variables
- Must remain lightweight and simple

Notes:
- This assumes a single-process server (1 training at a time)
- Not suitable for multi-worker or distributed setups
======================================================
"""

# --------------------------------------------------
# Training runtime state
# --------------------------------------------------

# Active training subprocess (Popen instance)
train_process = None

# Name of the currently running training job
active_run = None
