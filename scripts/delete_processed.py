from pathlib import Path
import shutil

def clear_processed():
    folder = Path("data/processed")

    if not folder.exists():
        print("Directory does not exist.")
        return

    for item in folder.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()          # delete file
        elif item.is_dir():
            shutil.rmtree(item)   # delete folder recursively

    print("data/processed cleared.")

if __name__ == "__main__":
    clear_processed()