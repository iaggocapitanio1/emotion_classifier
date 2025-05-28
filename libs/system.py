from pathlib import Path
from typing import List


def get_files(directory: Path) -> List[Path]:
    """
    Return a list of all files (not directories) in `directory`.
    Raises NotADirectoryError if the path isn't a directory.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(f"{directory.__str__()} is not a directory")
    return [p for p in directory.iterdir() if p.is_file()]

def get_dirs(directory: Path) -> List[Path]:
    """
    Return a list of all directories in `directory`.
    :param directory:
    :return:
    """
    directory = Path(directory)

    if not directory.is_dir():
        raise NotADirectoryError(f"{directory.__str__()} is not a directory")
    return [d for d in directory.iterdir() if d.is_dir()]