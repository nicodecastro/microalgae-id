"""Launch the offline desktop image-review prototype from this checkout."""

from pathlib import Path
import sys


def main() -> int:
    source_dir = Path(__file__).resolve().parent / "src"
    sys.path.insert(0, str(source_dir))
    from microalgae.ui.app import run

    return run()


if __name__ == "__main__":
    raise SystemExit(main())
