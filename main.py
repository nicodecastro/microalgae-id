from pathlib import Path
import argparse
import sys


def main(argv=None):
    # Support running directly from the checkout without an editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from training.config import load_config
    from training.experiments.runner import run_configured_experiment

    parser = argparse.ArgumentParser(description="Fresh extraction and nested-CV experiment (no holdout)")
    parser.add_argument("--config", type=Path, help="Training YAML (default: configs/training.yaml)")
    parser.add_argument("--output-dir", type=Path, required=True, help="New run directory; no overwrite")
    args = parser.parse_args(argv)
    try:
        result = run_configured_experiment(load_config(args.config), args.output_dir)
    except Exception as error:
        print(f"Experiment failed: {error}", file=sys.stderr)
        return 1
    print(f"Experiment completed: {result['output_dir']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
