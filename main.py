from pathlib import Path
import sys


def main():
    # Support running directly from the checkout without an editable install.
    sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
    from training.experiments.exp_sample import run_experiment

    run_experiment()


if __name__ == "__main__":
    main()
