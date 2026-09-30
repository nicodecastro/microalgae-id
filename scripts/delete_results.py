"""Delete generated contents of the repository's results directory."""

from pathlib import Path
import shutil


def clear_results():
    results_dir = Path(__file__).resolve().parent.parent / "results"

    if not results_dir.exists():
        print(f"Directory does not exist: {results_dir}")
        return

    for item in results_dir.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)

    print(f"Cleared: {results_dir}")


if __name__ == "__main__":
    clear_results()
