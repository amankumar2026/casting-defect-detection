# Evaluation results

Run on 2026-10-10 by `python eval/run_eval.py`. Test set: 260 original photos (156 defective, 104 ok) that were not used for training or for choosing thresholds.

## 1. Test results at the cost-based threshold

| Model | Threshold | Accuracy (95% CI) | Defects caught (95% CI) | Missed defects | False rejects | ROC AUC |
|---|---|---|---|---|---|---|
| Small CNN (from scratch) | 0.16 | 92.3% (88.4% to 95.0%) | 153/156 = 98.1% (94.5% to 99.3%) | 3 | 17 | 0.9885 |
| ResNet-18 (pretrained) | 0.09 | 98.8% (96.7% to 99.6%) | 155/156 = 99.4% (96.5% to 99.9%) | 1 | 2 | 0.9997 |
| ResNet-18 + lighting augmentation | 0.52 | 99.2% (97.2% to 99.8%) | 154/156 = 98.7% (95.4% to 99.6%) | 2 | 0 | 0.9989 |

## 2. Default threshold (0.5) against the cost-based threshold

Cost = 10 per missed defect + 1 per false reject. The threshold is chosen on the validation photos.

| Model | Threshold | Missed defects | False rejects | Cost |
|---|---|---|---|---|
| Small CNN (from scratch) | 0.50 | 10 | 8 | 108 |
| Small CNN (from scratch) | 0.16 | 3 | 17 | 47 |
| ResNet-18 (pretrained) | 0.50 | 2 | 0 | 20 |
| ResNet-18 (pretrained) | 0.09 | 1 | 2 | 12 |
| ResNet-18 + lighting augmentation | 0.50 | 2 | 0 | 20 |
| ResNet-18 + lighting augmentation | 0.52 | 2 | 0 | 20 |

## 3. How many labelled photos are needed?

| Model | Training photos | Accuracy | Defects caught | Missed defects | False rejects | ROC AUC |
|---|---|---|---|---|---|---|
| Small CNN (from scratch) | 78 | 93.1% | 95.5% | 7 | 11 | 0.9769 |
| Small CNN (from scratch) | 780 | 92.3% | 98.1% | 3 | 17 | 0.9885 |
| ResNet-18 (pretrained) | 78 | 93.8% | 98.7% | 2 | 14 | 0.9938 |
| ResNet-18 (pretrained) | 780 | 98.8% | 99.4% | 1 | 2 | 0.9997 |

## 4. Photo conditions not seen in training

Accuracy, with the share of defects caught in brackets.

| Condition | Small CNN (from scratch) | ResNet-18 (pretrained) | ResNet-18 + lighting augmentation |
|---|---|---|---|
| Clean test photos | 92.3% (98.1%) | 98.8% (99.4%) | 99.2% (98.7%) |
| Darker (60% brightness) | 61.9% (99.4%) | 95.0% (98.7%) | 98.8% (98.1%) |
| Brighter (130% brightness) | 86.5% (90.4%) | 97.7% (98.1%) | 99.2% (98.7%) |
| Out of focus (blur) | 59.2% (98.7%) | 80.8% (100.0%) | 91.5% (93.6%) |
| Sensor noise | 51.2% (76.9%) | 76.2% (99.4%) | 82.7% (95.5%) |
| Low resolution (96 px) | 69.2% (92.3%) | 81.9% (100.0%) | 91.2% (98.7%) |

The same runs as a cost, with missed defects + false rejects in brackets.

| Condition | Small CNN (from scratch) | ResNet-18 (pretrained) | ResNet-18 + lighting augmentation |
|---|---|---|---|
| Clean test photos | 47 (3 + 17) | 12 (1 + 2) | 20 (2 + 0) |
| Darker (60% brightness) | 108 (1 + 98) | 31 (2 + 11) | 30 (3 + 0) |
| Brighter (130% brightness) | 170 (15 + 20) | 33 (3 + 3) | 20 (2 + 0) |
| Out of focus (blur) | 124 (2 + 104) | 50 (0 + 50) | 112 (10 + 12) |
| Sensor noise | 451 (36 + 91) | 71 (1 + 61) | 108 (7 + 38) |
| Low resolution (96 px) | 188 (12 + 68) | 47 (0 + 47) | 41 (2 + 21) |

## 5. Speed on CPU

| Model | Time per photo | Training time |
|---|---|---|
| Small CNN (from scratch) | 13 ms | 13.2 min |
| ResNet-18 (pretrained) | 91 ms | 21.0 min |
| ResNet-18 + lighting augmentation | 93 ms | 21.7 min |

## 6. Shortcut check: can lighting alone predict the label?

A logistic regression on three numbers per photo (average brightness, contrast, background brightness). It cannot see any defect.

| Model | Accuracy | ROC AUC |
|---|---|---|
| Always answer "defective" | 60.0% | 0.5000 |
| Lighting only | 78.1% | 0.8600 |

Average brightness (0 to 255): defective photos 139, ok photos 149.

## Mistakes on the test set

- Small CNN (from scratch): `cast_def_0_2483.jpeg`, missed defect, p(defective) = 0.07
- Small CNN (from scratch): `cast_def_0_7581.jpeg`, missed defect, p(defective) = 0.11
- Small CNN (from scratch): `cast_def_0_9507.jpeg`, missed defect, p(defective) = 0.16
- Small CNN (from scratch): `cast_ok_0_1192.jpeg`, false reject, p(defective) = 0.75
- Small CNN (from scratch): `cast_ok_0_1392.jpeg`, false reject, p(defective) = 0.71
- Small CNN (from scratch): `cast_ok_0_1634.jpeg`, false reject, p(defective) = 0.18
- Small CNN (from scratch): `cast_ok_0_2505.jpeg`, false reject, p(defective) = 0.70
- Small CNN (from scratch): `cast_ok_0_3136.jpeg`, false reject, p(defective) = 0.35
- Small CNN (from scratch): `cast_ok_0_3199.jpeg`, false reject, p(defective) = 0.32
- Small CNN (from scratch): `cast_ok_0_3820.jpeg`, false reject, p(defective) = 0.24
- Small CNN (from scratch): `cast_ok_0_5191.jpeg`, false reject, p(defective) = 0.22
- Small CNN (from scratch): `cast_ok_0_6160.jpeg`, false reject, p(defective) = 0.43
- Small CNN (from scratch): `cast_ok_0_6890.jpeg`, false reject, p(defective) = 0.18
- Small CNN (from scratch): `cast_ok_0_7609.jpeg`, false reject, p(defective) = 0.31
- Small CNN (from scratch): `cast_ok_0_7673.jpeg`, false reject, p(defective) = 0.77
- Small CNN (from scratch): `cast_ok_0_850.jpeg`, false reject, p(defective) = 0.93
- Small CNN (from scratch): `cast_ok_0_904.jpeg`, false reject, p(defective) = 0.67
- Small CNN (from scratch): `cast_ok_0_9202.jpeg`, false reject, p(defective) = 0.32
- Small CNN (from scratch): `cast_ok_0_9367.jpeg`, false reject, p(defective) = 0.53
- Small CNN (from scratch): `cast_ok_0_9393.jpeg`, false reject, p(defective) = 0.71
- ResNet-18 (pretrained): `cast_def_0_2483.jpeg`, missed defect, p(defective) = 0.04
- ResNet-18 (pretrained): `cast_ok_0_6802.jpeg`, false reject, p(defective) = 0.24
- ResNet-18 (pretrained): `cast_ok_0_850.jpeg`, false reject, p(defective) = 0.18
- ResNet-18 + lighting augmentation: `cast_def_0_2483.jpeg`, missed defect, p(defective) = 0.10
- ResNet-18 + lighting augmentation: `cast_def_0_571.jpeg`, missed defect, p(defective) = 0.01
