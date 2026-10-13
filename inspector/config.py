"""All settings in one place."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

# Data
KAGGLE_DATASET = "ravirajsinh45/real-life-industrial-dataset-of-casting-product"
IMAGE_SIZE = 224       # every photo is resized to 224 x 224 grayscale
VAL_FRACTION = 0.2
TEST_FRACTION = 0.2
SEED = 42

# Training
MODELS = ("small_cnn", "resnet18", "resnet18_relit")
BATCH_SIZE = 32
EPOCHS = {"small_cnn": 40, "resnet18": 8, "resnet18_relit": 8}
LEARNING_RATE = {"small_cnn": 2e-3, "resnet18": 1e-4, "resnet18_relit": 1e-4}   # starting rate; decays to zero
# How much each training photo's brightness is randomly changed (1.0 = unchanged).
# resnet18_relit is the same model as resnet18, trained with a much wider range.
LIGHTING_RANGE = {"small_cnn": (0.9, 1.1), "resnet18": (0.9, 1.1), "resnet18_relit": (0.5, 1.5)}
LABEL_FRACTIONS = (0.1, 1.0)         # shares of the training photos to compare

# Business costs, in units of "one good part scrapped by mistake".
# A defective part that reaches the customer is assumed to cost 10 times more.
COST_FALSE_REJECT = 1
COST_MISSED_DEFECT = 10
