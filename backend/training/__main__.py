"""Entry point for ``python -m backend.training``.

Delegates directly to ``cli.main()`` for all subcommands.
"""

from backend.training.cli import main

if __name__ == "__main__":
    main()
