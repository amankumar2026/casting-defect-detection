"""Model plumbing and Grad-CAM, checked on untrained models and random photos.
No dataset, trained weights or download is needed."""

import numpy as np
import pytest
import torch

from inspector import config
from inspector.gradcam import grad_cam, overlay
from inspector.models import build_model, model_path, predict_proba

SIZE = config.IMAGE_SIZE


@pytest.fixture(scope="module")
def photos():
    return np.random.default_rng(0).integers(0, 255, (3, SIZE, SIZE), dtype=np.uint8)


@pytest.fixture(scope="module", params=config.MODELS)
def model(request):
    torch.manual_seed(0)
    return build_model(request.param, pretrained=False).eval()


def test_model_returns_one_score_per_photo(model):
    assert model(torch.rand(3, 1, SIZE, SIZE)).shape == (3,)


def test_predict_proba_returns_probabilities(model, photos):
    probs = predict_proba(model, photos, batch_size=2)
    assert probs.shape == (3,)
    assert ((probs >= 0) & (probs <= 1)).all()


def test_predict_proba_does_not_depend_on_batch_size(model, photos):
    assert predict_proba(model, photos, batch_size=1) == pytest.approx(
        predict_proba(model, photos, batch_size=3), abs=1e-5)


def test_grad_cam_is_a_photo_sized_map_from_zero_to_one(model, photos):
    heatmap = grad_cam(model, photos[0])
    assert heatmap.shape == (SIZE, SIZE)
    assert heatmap.min() >= 0 and heatmap.max() <= 1


def test_grad_cam_removes_its_hook(model, photos):
    grad_cam(model, photos[0])
    assert len(model.cam_layer._forward_hooks) == 0


def test_overlay_only_changes_pixels_under_the_heatmap():
    photo = np.full((4, 4), 100, dtype=np.uint8)
    heatmap = np.zeros((4, 4), dtype=np.float32)
    heatmap[0, 0] = 1.0
    image = overlay(photo, heatmap, strength=0.5)
    assert image.shape == (4, 4, 3) and image.dtype == np.uint8
    assert image[1, 1].tolist() == [100, 100, 100]
    assert image[0, 0, 0] > 100 > image[0, 0, 1]      # redder where the heatmap is hot


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError):
        build_model("vgg")


def test_models_trained_on_less_data_get_their_own_file():
    assert model_path("resnet18").name == "resnet18.pt"
    assert model_path("resnet18", 0.1).name == "resnet18_10pct.pt"
