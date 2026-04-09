---
comments: true
description: Master knowledge distillation for Ultralytics YOLO to optimize student model performance with a teacher model.
keywords: knowledge distillation, YOLO, Ultralytics, object detection, machine learning, computer vision, distill_model, teacher model, student model
---

# Knowledge Distillation

## What is Knowledge Distillation?

[Knowledge distillation](https://www.ultralytics.com/glossary/knowledge-distillation) is a technique where a compact student model learns from a larger, well-trained teacher model by imitating its behavior. It can enable the student model to achieve higher performance than training it directly.

## Preparing for Knowledge Distillation

Before starting distillation training, you need a **pre-trained teacher model**. The teacher model must be:

- A larger, more accurate model from the **same YOLO generation** as the student model. For example, you can distill YOLO26m into YOLO26n, but you cannot use a YOLO11 model as the teacher for a YOLO26 student.
- Already trained on the same target data and task as the student model.

!!! note

    Knowledge distillation currently supports **detect** tasks only.

### How It Works

1. A **teacher model**, which is larger and already trained on the target task, is frozen and used only for inference to guide the training of the student model.
2. A smaller **student model** is optimized for the same target task using the standard training losses, while also being guided by the teacher through feature-level distillation.
3. At each training iteration, intermediate features are extracted from both models at automatically determined layers.
4. A **projector network** (lightweight MLP) align the student's feature dimensions to match the teacher's.
5. A **score-weighted L2 loss** compares the projected student features with the teacher features, where the loss is weighted by the teacher's classification scores.
6. The distillation loss is combined with the standard loss using a configurable weight.

## Key Parameters

| Parameter      | Type    | Default | Description                                                                                                          |
| -------------- | ------- | ------- | -------------------------------------------------------------------------------------------------------------------- |
| `distill_model`| `str`   | `None`  | Path to the teacher model file (e.g., `yolo26x.pt`). Setting this parameter enables knowledge distillation.          |
| `dis`          | `float` | `6`   | Distillation loss weight. Controls how much the distillation loss contributes to the total training loss.             |

## Train

Training with knowledge distillation is nearly identical to standard training. The only difference is providing the `distill_model` argument, which specifies the path to a pre-trained teacher model.

!!! example "Knowledge Distillation Training"

    === "Python"

        ```python
        from ultralytics import YOLO

        # Load a student model
        student = YOLO("yolo26m.pt")

        # Train with knowledge distillation from a larger teacher model
        results = student.train(
            data="coco8.yaml",
            epochs=100,
            distill_model="yolo26x.pt",  # path to teacher model
        )
        ```

    === "CLI"

        ```bash
        yolo detect train model=yolo26m.pt data=coco8.yaml epochs=100 distill_model=yolo26x.pt
        ```

### Adjusting the Distillation Loss Weight

The `dis` parameter controls how much the distillation loss influences training. A higher value places more emphasis on mimicking the teacher, while a lower value lets the standard detection loss dominate.

!!! example "Custom Distillation Weight"

    === "Python"

        ```python
        from ultralytics import YOLO

        student = YOLO("yolo26m.pt")

        results = student.train(
            data="coco8.yaml",
            epochs=100,
            distill_model="yolo26x.pt",
            dis=10,  # increase distillation loss weight
        )
        ```

    === "CLI"

        ```bash
        yolo detect train model=yolo26m.pt data=coco8.yaml epochs=100 distill_model=yolo26x.pt dis=10
        ```

### Resuming Distillation Training

Distillation training supports resuming from a checkpoint. The teacher model state is automatically restored from the saved checkpoint.

!!! example "Resume Distillation Training"

    === "Python"

        ```python
        from ultralytics import YOLO

        student = YOLO("runs/detect/train/weights/last.pt")

        results = student.train(
            resume=True,
        )
        ```

    === "CLI"

        ```bash
        yolo detect train resume model=runs/detect/train/weights/last.pt
        ```

## Training Output

When distillation is enabled, an additional `dis_loss` column appears in the training logs alongside the standard loss components:

```
      Epoch    GPU_mem   box_loss   cls_loss   dfl_loss   dis_loss  Instances       Size
      1/80      46.2G      1.566      5.404    0.003249      6.658        231        640
```

The final exported model contains **only the student weights**, so the model file size and inference speed are identical to a normally trained student model.

## FAQ

### What models can I use for knowledge distillation?

Currently, knowledge distillation supports **detect** task models only. The teacher model is typically larger than the student, but using the largest teacher does not always lead to the best performance. The following combinations are recommended:

| Student         | Teacher         |
| --------------- | --------------- |
| `yolo26n.pt`    | `yolo26s.pt`    |
| `yolo26s.pt`    | `yolo26m.pt`    |
| `yolo26m.pt`    | `yolo26x.pt`    |
| `yolo26l.pt`    | `yolo26x.pt`    |

Cross-generation distillation (e.g., YOLO11 teacher with a YOLO26 student) is **not supported**.

### How does distillation differ from standard training?

The only difference is the addition of the `distill_model` parameter. Everything else works the same way. During training, an extra distillation loss is computed and added to the total loss, but the final saved model is a standard YOLO model with no extra overhead.

### What is the `dis` parameter and how should I set it?

The `dis` parameter (default `6`) controls the weight of the distillation loss relative to the standard losses. Start with the default value, and adjust based on your results:

- **Increase** `dis` (e.g., `10`) if the student model is significantly smaller and needs more guidance from the teacher.
- **Decrease** `dis` (e.g., `1`) if the distillation loss is dominating and hurting detection performance.

### Does knowledge distillation slow down training?

Yes, there is a moderate increase in training time and GPU memory usage because the teacher model runs inference on each batch (in eval mode, with no gradient computation). However, the teacher forward pass is efficient since no gradients are tracked, and the overall overhead is manageable on modern GPUs.
