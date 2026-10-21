"""Export configured feature families to a fresh directory, without training."""

import argparse
from pathlib import Path
import sys


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from training.config import load_config
from training.extraction import extract_configured_features


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="Training YAML (default: configs/training.yaml)")
    parser.add_argument("--output-dir", type=Path, required=True, help="New directory; existing paths are rejected")
    args = parser.parse_args(argv)
    try:
        frames = extract_configured_features(load_config(args.config), args.output_dir)
    except Exception as error:
        print(f"Extraction failed: {error}", file=sys.stderr)
        return 1
    print("Extraction and audit passed: " + ", ".join(f"{name} ({len(frame)} images)" for name, frame in frames.items()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
