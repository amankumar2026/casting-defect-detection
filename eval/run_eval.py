"""Evaluation pipeline. Run after training:  python eval/run_eval.py

Every number comes from the 260 test photos, which are never used for
training or for choosing a threshold. Writes eval/results.md,
eval/results.json and two figures in reports/.
"""

import json
import sys
import time
from datetime import date
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.v2.functional as TF
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inspector import config
from inspector.data import SPLIT_FILE, load_split, to_tensor
from inspector.gradcam import grad_cam, overlay
from inspector.metrics import pick_threshold, summarize, wilson_interval
from inspector.models import load_model, model_path, predict_proba

EVAL_DIR = Path(__file__).resolve().parent
NAMES = {"small_cnn": "Small CNN (from scratch)", "resnet18": "ResNet-18 (pretrained)",
         "resnet18_relit": "ResNet-18 + lighting augmentation"}


# --- photo conditions the models never saw in training -----------------------

def darker(images):
    return (images * 0.6).astype(np.uint8)

def brighter(images):
    return (images * 1.3).clip(0, 255).astype(np.uint8)

def blurred(images):
    out = TF.gaussian_blur(to_tensor(images), kernel_size=9, sigma=2.0)
    return (out[:, 0] * 255).round().byte().numpy()

def noisy(images):
    noise = np.random.default_rng(config.SEED).normal(0, 20, images.shape)
    return (images + noise).clip(0, 255).astype(np.uint8)

def low_resolution(images):
    small = TF.resize(to_tensor(images), [96, 96], antialias=True)
    out = TF.resize(small, [config.IMAGE_SIZE, config.IMAGE_SIZE], antialias=True)
    return (out[:, 0].clamp(0, 1) * 255).round().byte().numpy()

CONDITIONS = {
    "Clean test photos": lambda images: images,
    "Darker (60% brightness)": darker,
    "Brighter (130% brightness)": brighter,
    "Out of focus (blur)": blurred,
    "Sensor noise": noisy,
    "Low resolution (96 px)": low_resolution,
}


# --- measurements ------------------------------------------------------------

def milliseconds_per_photo(model, images) -> float:
    """Average time to score one photo at a time on this CPU."""
    predict_proba(model, images[:2])            # warm-up
    started = time.perf_counter()
    for image in images[:40]:
        predict_proba(model, image[None])
    return (time.perf_counter() - started) / 40 * 1000


def evaluate_model(name: str, fraction: float, val, test) -> dict:
    model = load_model(name, fraction)
    threshold = pick_threshold(val[1], predict_proba(model, val[0]))
    probs = predict_proba(model, test[0])
    at_threshold = summarize(test[1], probs, threshold)
    defects = int(test[1].sum())
    return {
        "model": name, "fraction": fraction, "threshold": threshold,
        "auc": float(roc_auc_score(test[1], probs)),
        "at_0.5": summarize(test[1], probs, 0.5),
        "at_threshold": at_threshold,
        "accuracy_ci": wilson_interval(round(at_threshold["accuracy"] * len(test[1])), len(test[1])),
        "recall_ci": wilson_interval(defects - at_threshold["missed_defects"], defects),
        "probs": probs,
    }


def brightness_features(images: np.ndarray) -> np.ndarray:
    """Three numbers per photo that say nothing about defects: average
    brightness, contrast, and the brightness of the background corners."""
    corners = np.concatenate([images[:, :20, :20], images[:, :20, -20:],
                              images[:, -20:, :20], images[:, -20:, -20:]], axis=1)
    return np.column_stack([images.mean(axis=(1, 2)), images.std(axis=(1, 2)),
                            corners.mean(axis=(1, 2))])


def brightness_baseline(train, test) -> dict:
    """Can the label be guessed from lighting alone? If yes, the photos contain
    a shortcut that a CNN could learn in place of the defects."""
    model = make_pipeline(StandardScaler(), LogisticRegression())
    model.fit(brightness_features(train[0]), train[1])
    probs = model.predict_proba(brightness_features(test[0]))[:, 1]
    means = brightness_features(np.concatenate([train[0], test[0]]))[:, 0]
    labels = np.concatenate([train[1], test[1]])
    return {"accuracy": float(((probs >= 0.5) == test[1]).mean()),
            "auc": float(roc_auc_score(test[1], probs)),
            "majority_class_accuracy": float(max(test[1].mean(), 1 - test[1].mean())),
            "mean_brightness_defective": float(means[labels == 1].mean()),
            "mean_brightness_ok": float(means[labels == 0].mean())}


def robustness(name: str, threshold: float, test) -> dict:
    model = load_model(name)
    return {label: summarize(test[1], predict_proba(model, change(test[0])), threshold)
            for label, change in CONDITIONS.items()}


# --- figures -----------------------------------------------------------------

def gradcam_figure(results: dict, test, path: Path) -> None:
    """The first six defective test photos in file order (not hand-picked),
    with each model's heatmap."""
    rows = np.flatnonzero(test[1] == 1)[:6]
    models = {name: load_model(name) for name in config.MODELS}
    lines = 1 + len(config.MODELS)
    figure, axes = plt.subplots(lines, 6, figsize=(15, 2.6 * lines))
    for column, row in enumerate(rows):
        photo = test[0][row]
        axes[0, column].imshow(photo, cmap="gray", vmin=0, vmax=255)
        for line, name in enumerate(config.MODELS, start=1):
            axes[line, column].imshow(overlay(photo, grad_cam(models[name], photo), 0.6))
            axes[line, column].set_title(f"p(defective) = {results[name]['probs'][row]:.2f}",
                                         fontsize=9)
    for line, label in enumerate(["Photo", *[NAMES[n] for n in config.MODELS]]):
        axes[line, 0].set_ylabel(label, fontsize=10)
    for axis in axes.ravel():
        axis.set_xticks([]); axis.set_yticks([])
    figure.suptitle("Grad-CAM on defective test parts: red = what pushed the model towards 'defective'")
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


def curves_figure(path: Path) -> None:
    figure, axes = plt.subplots(1, len(config.MODELS), figsize=(5 * len(config.MODELS), 3.6))
    for axis, name in zip(axes, config.MODELS):
        history = pd.DataFrame(json.loads(model_path(name).with_suffix(".json").read_text())["history"])
        axis.plot(history["epoch"], history["train_loss"], label="Training loss")
        axis.plot(history["epoch"], history["val_loss"], label="Validation loss")
        axis.set_title(NAMES[name]); axis.set_xlabel("Epoch"); axis.set_yscale("log")
        axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(path, dpi=110)
    plt.close(figure)


# --- report ------------------------------------------------------------------

def pct(value: float) -> str:
    return f"{value:.1%}"


def write_report(results: dict, efficiency: list, robust: dict, speed: dict,
                 shortcut: dict, test) -> None:
    n, defects = len(test[1]), int(test[1].sum())
    lines = [
        "# Evaluation results", "",
        f"Run on {date.today().isoformat()} by `python eval/run_eval.py`. "
        f"Test set: {n} original photos ({defects} defective, {n - defects} ok) "
        "that were not used for training or for choosing thresholds.", "",
        "## 1. Test results at the cost-based threshold", "",
        "| Model | Threshold | Accuracy (95% CI) | Defects caught (95% CI) | Missed defects | False rejects | ROC AUC |",
        "|---|---|---|---|---|---|---|",
    ]
    for name in config.MODELS:
        r, s = results[name], results[name]["at_threshold"]
        lines.append(
            f"| {NAMES[name]} | {r['threshold']:.2f} "
            f"| {pct(s['accuracy'])} ({pct(r['accuracy_ci'][0])} to {pct(r['accuracy_ci'][1])}) "
            f"| {defects - s['missed_defects']}/{defects} = {pct(s['recall'])} "
            f"({pct(r['recall_ci'][0])} to {pct(r['recall_ci'][1])}) "
            f"| {s['missed_defects']} | {s['false_rejects']} | {r['auc']:.4f} |")

    lines += ["", "## 2. Default threshold (0.5) against the cost-based threshold", "",
              f"Cost = {config.COST_MISSED_DEFECT} per missed defect + "
              f"{config.COST_FALSE_REJECT} per false reject. The threshold is chosen on "
              "the validation photos.", "",
              "| Model | Threshold | Missed defects | False rejects | Cost |", "|---|---|---|---|---|"]
    for name in config.MODELS:
        for key in ("at_0.5", "at_threshold"):
            s = results[name][key]
            lines.append(f"| {NAMES[name]} | {s['threshold']:.2f} | {s['missed_defects']} "
                         f"| {s['false_rejects']} | {s['cost']} |")

    lines += ["", "## 3. How many labelled photos are needed?", "",
              "| Model | Training photos | Accuracy | Defects caught | Missed defects | False rejects | ROC AUC |",
              "|---|---|---|---|---|---|---|"]
    for r in efficiency:
        s = r["at_threshold"]
        lines.append(f"| {NAMES[r['model']]} | {r['train_photos']} | {pct(s['accuracy'])} "
                     f"| {pct(s['recall'])} | {s['missed_defects']} | {s['false_rejects']} "
                     f"| {r['auc']:.4f} |")

    lines += ["", "## 4. Photo conditions not seen in training", "",
              "Accuracy, with the share of defects caught in brackets.", "",
              "| Condition | " + " | ".join(NAMES[n] for n in config.MODELS) + " |",
              "|---|" + "---|" * len(config.MODELS)]
    for label in CONDITIONS:
        cells = [f"{pct(robust[n][label]['accuracy'])} ({pct(robust[n][label]['recall'])})"
                 for n in config.MODELS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["", "The same runs as a cost, with missed defects + false rejects in brackets.", "",
              "| Condition | " + " | ".join(NAMES[n] for n in config.MODELS) + " |",
              "|---|" + "---|" * len(config.MODELS)]
    for label in CONDITIONS:
        cells = [f"{robust[n][label]['cost']} ({robust[n][label]['missed_defects']} + "
                 f"{robust[n][label]['false_rejects']})" for n in config.MODELS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")

    lines += ["", "## 5. Speed on CPU", "",
              "| Model | Time per photo | Training time |", "|---|---|---|"]
    for name in config.MODELS:
        lines.append(f"| {NAMES[name]} | {speed[name]['ms_per_photo']:.0f} ms "
                     f"| {speed[name]['train_minutes']} min |")

    lines += ["", "## 6. Shortcut check: can lighting alone predict the label?", "",
              "A logistic regression on three numbers per photo (average brightness, "
              "contrast, background brightness). It cannot see any defect.", "",
              "| Model | Accuracy | ROC AUC |", "|---|---|---|",
              f"| Always answer \"defective\" | {pct(shortcut['majority_class_accuracy'])} | 0.5000 |",
              f"| Lighting only | {pct(shortcut['accuracy'])} | {shortcut['auc']:.4f} |", "",
              f"Average brightness (0 to 255): defective photos "
              f"{shortcut['mean_brightness_defective']:.0f}, ok photos "
              f"{shortcut['mean_brightness_ok']:.0f}."]

    lines += ["", "## Mistakes on the test set", ""]
    files = pd.read_csv(SPLIT_FILE).query("split == 'test'")["file"].to_numpy()
    for name in config.MODELS:
        r = results[name]
        wrong = np.flatnonzero((r["probs"] >= r["threshold"]) != test[1])
        if len(wrong) == 0:
            lines.append(f"- {NAMES[name]}: none.")
        for i in wrong:
            kind = "missed defect" if test[1][i] == 1 else "false reject"
            lines.append(f"- {NAMES[name]}: `{files[i]}`, {kind}, p(defective) = {r['probs'][i]:.2f}")

    (EVAL_DIR / "results.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    torch.set_num_threads(4)
    val, test = load_split("val"), load_split("test")

    results = {name: evaluate_model(name, 1.0, val, test) for name in config.MODELS}

    efficiency = []
    for name in ("small_cnn", "resnet18"):
        for fraction in config.LABEL_FRACTIONS:
            run_file = model_path(name, fraction).with_suffix(".json")   # written when training ends
            if run_file.exists():
                r = evaluate_model(name, fraction, val, test)
                r["train_photos"] = json.loads(run_file.read_text())["train_photos"]
                efficiency.append(r)

    robust = {name: robustness(name, results[name]["threshold"], test) for name in config.MODELS}
    speed = {name: {"ms_per_photo": milliseconds_per_photo(load_model(name), test[0]),
                    "train_minutes": json.loads(
                        model_path(name).with_suffix(".json").read_text())["minutes"]}
             for name in config.MODELS}

    config.REPORT_DIR.mkdir(exist_ok=True)
    gradcam_figure(results, test, config.REPORT_DIR / "gradcam_examples.png")
    curves_figure(config.REPORT_DIR / "training_curves.png")
    shortcut = brightness_baseline(load_split("train"), test)
    write_report(results, efficiency, robust, speed, shortcut, test)

    def without_probs(r):
        return {k: v for k, v in r.items() if k != "probs"}
    (EVAL_DIR / "results.json").write_text(json.dumps({
        "test": {name: without_probs(r) for name, r in results.items()},
        "label_efficiency": [without_probs(r) for r in efficiency],
        "robustness": robust, "speed": speed, "shortcut_check": shortcut}, indent=2))
    print((EVAL_DIR / "results.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
