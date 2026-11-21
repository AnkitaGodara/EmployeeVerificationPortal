"""The 4-layer CNN (same architecture as the original notebook) plus the fixes:

* Rescaling(1/255) is INSIDE the model, so training and prediction are always normalised the
  same way (the original notebook normalised only at prediction time).
* Augmentation (flip, rotation, zoom, shift) is inside the model too; Keras switches it off
  automatically at prediction time.
* Evaluation reports accuracy, precision, recall, F1 and a confusion matrix.
"""
import json
import os
from datetime import datetime, timezone

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from ..config import settings

MODEL_PATH = settings.MODEL_DIR / "face_cnn.keras"
LABELS_PATH = settings.MODEL_DIR / "labels.json"
METRICS_PATH = settings.MODEL_DIR / "metrics.json"


def build_model(num_classes: int, img_size: int = None):
    import keras
    from keras import layers
    img_size = img_size or settings.IMG_SIZE
    model = keras.Sequential([
        keras.Input(shape=(img_size, img_size, 3)),
        layers.Rescaling(1.0 / 255),                       # normalisation 0-255 -> 0-1
        layers.RandomFlip("horizontal"),                   # augmentation (training only)
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.1),
        layers.RandomTranslation(0.08, 0.08),
        layers.Conv2D(32, (4, 4), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (4, 4), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (4, 4), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (4, 4), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(64, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def train(images: np.ndarray, labels: list[str], epochs: int = None, progress=None) -> dict:
    """Train on face crops, evaluate on a held-out 20% split, save model + labels + metrics."""
    import keras
    epochs = epochs or settings.EPOCHS

    counts = {c: labels.count(c) for c in sorted(set(labels))}
    keep = {c for c, n in counts.items() if n >= settings.MIN_IMAGES_PER_PERSON}
    dropped = sorted(set(counts) - keep)
    if len(keep) < 2:
        raise ValueError(f"Need at least 2 people with {settings.MIN_IMAGES_PER_PERSON}+ face "
                         f"images each. Current counts: {counts}")
    mask = np.array([lab in keep for lab in labels])
    images = images[mask]
    labels = [lab for lab in labels if lab in keep]

    class_names = sorted(keep)
    y = np.array([class_names.index(lab) for lab in labels])
    x_train, x_val, y_train, y_val = train_test_split(
        images, y, test_size=0.2, stratify=y, random_state=42)

    weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
    class_weight = {int(c): float(w) for c, w in zip(np.unique(y_train), weights)}

    keras.utils.set_random_seed(42)
    model = build_model(len(class_names))
    callbacks = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,
                                               restore_best_weights=True)]
    if progress:
        callbacks.append(keras.callbacks.LambdaCallback(
            on_epoch_end=lambda epoch, logs: progress(epoch + 1, epochs, logs or {})))
    history = model.fit(x_train.astype("float32"), y_train,
                        validation_data=(x_val.astype("float32"), y_val),
                        epochs=epochs, batch_size=settings.BATCH_SIZE,
                        class_weight=class_weight, callbacks=callbacks, verbose=0)

    probs = model.predict(x_val.astype("float32"), verbose=0)
    y_pred = probs.argmax(axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average="macro", zero_division=0)
    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "classes": class_names,
        "dropped_classes": dropped,
        "num_images": int(len(images)),
        "train_images": int(len(x_train)),
        "val_images": int(len(x_val)),
        "epochs_run": len(history.history["loss"]),
        "accuracy": round(float(accuracy_score(y_val, y_pred)), 4),
        "precision_macro": round(float(precision), 4),
        "recall_macro": round(float(recall), 4),
        "f1_macro": round(float(f1), 4),
        "per_class": classification_report(y_val, y_pred, labels=list(range(len(class_names))),
                                           target_names=class_names, output_dict=True,
                                           zero_division=0),
        "confusion_matrix": confusion_matrix(y_val, y_pred,
                                             labels=list(range(len(class_names)))).tolist(),
        "history": {k: [round(float(v), 4) for v in vals] for k, vals in history.history.items()},
        "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
    }
    model.save(MODEL_PATH)
    LABELS_PATH.write_text(json.dumps(class_names))
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))
    return metrics


def load():
    """Return (model, class_names) or (None, []) if no model has been trained yet."""
    if not (MODEL_PATH.exists() and LABELS_PATH.exists()):
        return None, []
    import keras
    return keras.models.load_model(MODEL_PATH), json.loads(LABELS_PATH.read_text())


def load_metrics() -> dict | None:
    return json.loads(METRICS_PATH.read_text()) if METRICS_PATH.exists() else None
