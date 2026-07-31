from pathlib import Path
import shutil

def clear_success_fail():
    folder = Path("data/failed")

    if not folder.exists():
        print("Directory does not exist.")
        return

    for item in folder.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()          # delete file
        elif item.is_dir():
            shutil.rmtree(item)   # delete folder recursively

    print("data/failed cleared.")
    
    folder = Path("data/success")

    if not folder.exists():
        print("Directory does not exist.")
        return

    for item in folder.iterdir():
        if item.is_file() or item.is_symlink():
            item.unlink()          # delete file
        elif item.is_dir():
            shutil.rmtree(item)   # delete folder recursively

    print("data/success cleared.")

if __name__ == "__main__":
    clear_success_fail()