"""Splitting and preprocessing, checked without downloading the dataset."""

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from inspector import config
from inspector.data import (CastingDataset, augment, make_split, prepare_image,
                            subsample, to_tensor)


@pytest.fixture
def files():
    """500 fake file names: 300 defective, 200 ok."""
    return pd.DataFrame({
        "folder": ["def_front"] * 300 + ["ok_front"] * 200,
        "file": [f"photo_{i}.jpeg" for i in range(500)],
        "label": [1] * 300 + [0] * 200,
    })


def test_split_uses_every_photo_exactly_once(files):
    split = make_split(files)
    assert sorted(split["file"]) == sorted(files["file"])
    assert split["file"].is_unique


def test_split_sizes_are_60_20_20(files):
    sizes = make_split(files)["split"].value_counts()
    assert (sizes["train"], sizes["val"], sizes["test"]) == (300, 100, 100)


def test_each_split_keeps_the_same_share_of_defects(files):
    share = make_split(files).groupby("split")["label"].mean()
    assert share.to_numpy() == pytest.approx(0.6, abs=0.01)


def test_split_is_the_same_every_time(files):
    pd.testing.assert_frame_equal(make_split(files), make_split(files))


def test_subsample_keeps_the_class_balance():
    labels = np.array([1] * 60 + [0] * 40)
    images = np.zeros((100, 4, 4), dtype=np.uint8)
    small_images, small_labels = subsample(images, labels, 0.25)
    assert len(small_images) == len(small_labels) == 25
    assert small_labels.sum() == 15


def test_subsample_with_fraction_one_returns_everything():
    labels = np.array([1, 0, 1, 0])
    images = np.zeros((4, 4, 4), dtype=np.uint8)
    assert subsample(images, labels, 1.0)[1] is labels


def test_prepare_image_turns_any_photo_into_the_model_format():
    colour_photo = Image.new("RGB", (512, 300), (200, 100, 50))
    prepared = prepare_image(colour_photo)
    assert prepared.shape == (config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert prepared.dtype == np.uint8


def test_to_tensor_scales_pixels_to_zero_one():
    tensor = to_tensor(np.array([[[0, 255]]], dtype=np.uint8))
    assert tensor.shape == (1, 1, 1, 2)
    assert tensor.flatten().tolist() == [0.0, 1.0]


def test_augment_keeps_shape_and_pixel_range():
    image = torch.rand(1, 64, 64)
    changed = augment(image)
    assert changed.shape == image.shape
    assert 0 <= changed.min() and changed.max() <= 1


def test_only_training_photos_are_augmented():
    images = np.random.default_rng(0).integers(0, 255, (2, 32, 32), dtype=np.uint8)
    labels = np.array([1, 0])
    fixed = CastingDataset(images, labels, train=False)
    assert torch.equal(fixed[0][0], fixed[0][0])
    assert not torch.equal(CastingDataset(images, labels, train=True)[0][0], fixed[0][0])
