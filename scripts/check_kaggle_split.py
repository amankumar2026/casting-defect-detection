"""Does the dataset's ready-made train / test split leak?

The dataset ships a 300 x 300 folder already split into train and test.
This script counts test photos that are byte-for-byte the same file as a
training photo. Any such photo has been seen during training, so a score
on that test folder is too optimistic.

    python scripts/check_kaggle_split.py
"""

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspector.data import dataset_dir


def file_hashes(folder: Path) -> dict[str, str]:
    """Maps each photo's MD5 hash to its class folder name."""
    return {hashlib.md5(path.read_bytes()).hexdigest(): path.parent.name
            for path in folder.rglob("*.jpeg")}


def main() -> None:
    root = dataset_dir() / "casting_data" / "casting_data"
    train, test = file_hashes(root / "train"), file_hashes(root / "test")
    shared = set(train) & set(test)
    same_label = sum(train[h] == test[h] for h in shared)

    print(f"Train photos: {len(train)}   Test photos: {len(test)}")
    print(f"Test photos identical to a training photo: {len(shared)} "
          f"({len(shared) / len(test):.1%} of the test folder)")
    print(f"Of those, with the same label in both folders: {same_label}")


if __name__ == "__main__":
    main()
