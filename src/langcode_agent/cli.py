"""Backward-compatible CLI entrypoint.

Runtime implementation lives in :mod:`langcode_agent.interfaces.cli`.
"""

from .interfaces.cli import *  # noqa: F401,F403
from .interfaces.cli import main


if __name__ == "__main__":
    main()
