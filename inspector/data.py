"""Photos -> train / validation / test arrays.

    1,300 original photos -> stratified 60 / 20 / 20 split (saved to data/split.csv)
                          -> resized to 224 x 224 grayscale, cached in one .npz file
                          -> training photos are randomly rotated, flipped and re-lit
"""

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2.functional as TF
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

from inspector import config

SPLIT_FILE = config.DATA_DIR / "split.csv"
CACHE_FILE = config.DATA_DIR / f"originals_{config.IMAGE_SIZE}.npz"
LABELS = {"ok_front": 0, "def_front": 1}     # 1 = defective


def dataset_dir() -> Path:
    """Folder of the Kaggle dataset. Downloaded once (100 MB), then read from cache."""
    import kagglehub
    return Path(kagglehub.dataset_download(config.KAGGLE_DATASET))


def originals_dir() -> Path:
    return dataset_dir() / "casting_512x512" / "casting_512x512"


def make_split(files: pd.DataFrame, seed: int = config.SEED) -> pd.DataFrame:
    """Adds a "split" column. Each split keeps the same share of defective parts."""
    rest, test = train_test_split(files, test_size=config.TEST_FRACTION,
                                  stratify=files["label"], random_state=seed)
    val_share = config.VAL_FRACTION / (1 - config.TEST_FRACTION)
    train, val = train_test_split(rest, test_size=val_share,
                                  stratify=rest["label"], random_state=seed)
    split = pd.concat([train.assign(split="train"), val.assign(split="val"),
                       test.assign(split="test")])
    return split.sort_values(["folder", "file"]).reset_index(drop=True)


def prepare_image(image: Image.Image) -> np.ndarray:
    """Any photo -> 224 x 224 grayscale uint8, the format the models are trained on."""
    size = (config.IMAGE_SIZE, config.IMAGE_SIZE)
    return np.asarray(image.convert("L").resize(size, Image.BILINEAR), dtype=np.uint8)


def build_cache() -> None:
    """Writes data/split.csv and the resized-image cache."""
    config.DATA_DIR.mkdir(exist_ok=True)
    folder = originals_dir()
    files = pd.DataFrame(
        [{"folder": name, "file": path.name, "label": label}
         for name, label in LABELS.items()
         for path in sorted((folder / name).glob("*.jpeg"))])
    split = make_split(files)
    images = np.stack([prepare_image(Image.open(folder / row.folder / row.file))
                       for row in split.itertuples()])
    split.to_csv(SPLIT_FILE, index=False)
    np.savez_compressed(CACHE_FILE, images=images)


def load_split(name: str) -> tuple[np.ndarray, np.ndarray]:
    """Returns (images, labels) for "train", "val" or "test".
    images: uint8 array (n, 224, 224). labels: 1 = defective, 0 = ok."""
    if not (SPLIT_FILE.exists() and CACHE_FILE.exists()):
        build_cache()
    split = pd.read_csv(SPLIT_FILE)
    rows = (split["split"] == name).to_numpy()
    return np.load(CACHE_FILE)["images"][rows], split.loc[rows, "label"].to_numpy().copy()


def subsample(images: np.ndarray, labels: np.ndarray, fraction: float,
              seed: int = config.SEED) -> tuple[np.ndarray, np.ndarray]:
    """A stratified fraction of the data, to test how many labelled photos are needed."""
    if fraction >= 1:
        return images, labels
    keep, _ = train_test_split(np.arange(len(labels)), train_size=fraction,
                               stratify=labels, random_state=seed)
    return images[keep], labels[keep]


def to_tensor(images: np.ndarray) -> torch.Tensor:
    """uint8 (n, H, W) -> float (n, 1, H, W) in the range 0 to 1."""
    return torch.from_numpy(images).float().div(255).unsqueeze(1)


def augment(image: torch.Tensor, lighting: tuple[float, float] = (0.9, 1.1)) -> torch.Tensor:
    """Random rotation, flip, brightness and contrast for one (1, H, W) image.

    The part is round and photographed from above, so any rotation or flip is
    still a realistic photo. The corners exposed by rotating are filled with
    the photo's own background grey. `lighting` is the range the brightness
    is multiplied by.
    """
    background = float(torch.stack([image[0, 0, 0], image[0, 0, -1],
                                    image[0, -1, 0], image[0, -1, -1]]).mean())
    image = TF.rotate(image, random.uniform(0, 360), fill=background,
                      interpolation=TF.InterpolationMode.BILINEAR)
    if random.random() < 0.5:
        image = TF.horizontal_flip(image)
    image = TF.adjust_brightness(image, random.uniform(*lighting))
    image = TF.adjust_contrast(image, random.uniform(0.9, 1.1))
    return image


class CastingDataset(Dataset):
    def __init__(self, images: np.ndarray, labels: np.ndarray, train: bool = False,
                 lighting: tuple[float, float] = (0.9, 1.1)):
        self.images = to_tensor(images)
        self.labels = torch.from_numpy(labels).float()
        self.train = train
        self.lighting = lighting

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int):
        image = self.images[i]
        return (augment(image, self.lighting) if self.train else image), self.labels[i]
