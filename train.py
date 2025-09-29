#!/usr/bin/env python3
import os
import sys


def main() -> None:
    # Ensure repository root is on sys.path so imports work when run from anywhere
    repo_root = os.path.dirname(os.path.abspath(__file__))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    from dna_bind_offline.train.cli import main as cli_main
    cli_main()


if __name__ == "__main__":
    main()


