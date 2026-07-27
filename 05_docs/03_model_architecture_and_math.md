# Stage 2 Documentation: Dual-Head Model Architecture & Mathematical Loss Formulation

## 1. Network Architecture

The architecture builds upon the **YOLOv11 Medium** backbone and neck (FPN + PAN), introducing a custom parallel output head (`CustomSegmentHead`):

```
Feature Map Input (P3, P4, P5 scales)
        │
        ├──► cv2 : Bounding Box Regression (DFL + CIoU)
        ├──► cv3 : Classification (BCE)
        ├──► cv4 : Mask Coefficients (32 channels)
        └──► cv5 : Root Keypoint Regression (2 channels: x, y)
```

## 2. Mathematical Loss Formulation

The multi-task objective function is defined as a weighted linear combination of 5 loss components:

$$ \mathcal{L}_{total} = \lambda_{box} \mathcal{L}_{box} + \lambda_{seg} \mathcal{L}_{seg} + \lambda_{cls} \mathcal{L}_{cls} + \lambda_{dfl} \mathcal{L}_{dfl} + \lambda_{kpt} \mathcal{L}_{kpt} $$

### Root Keypoint Regression Loss ($\mathcal{L}_{kpt}$)
For foreground matched anchors $i \in \mathcal{N}_{fg}$, the keypoint loss minimizes the L1 distance between predicted root coordinates $(\hat{x}_i, \hat{y}_i)$ and ground-truth root coordinates $(x_i, y_i)$:

$$ \mathcal{L}_{kpt} = \frac{1}{|\mathcal{N}_{fg}|} \sum_{i \in \mathcal{N}_{fg}} \left( |\hat{x}_i - x_i| + |\hat{y}_i - y_i| \right) $$

## 3. Keypoint Accuracy Evaluation Metric (PCK@10)

Keypoint performance is evaluated using **Percentage of Correct Keypoints (PCK)** normalized by bounding box max dimension $S = \max(W_{box}, H_{box})$:

$$ \text{PCK}@\alpha = \frac{1}{N} \sum_{i=1}^N \mathbb{I} \left( \frac{d(\mathbf{k}_i, \hat{\mathbf{k}}_i)}{S_i} \le \alpha \right) $$

where $\alpha = 0.10$ for PCK@10.
