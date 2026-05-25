"""Backward-compatible web entrypoint.

Runtime implementation lives in :mod:`langcode_agent.interfaces.web`.
"""

from .interfaces.web import *  # noqa: F401,F403
from .interfaces.web import main


if __name__ == "__main__":
    main()
