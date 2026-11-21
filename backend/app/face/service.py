"""Keeps the trained model in memory, runs training in a background thread and answers
"who is this?" for a face crop."""
import threading
import traceback
from datetime import datetime, timezone

import numpy as np

from ..config import settings
from ..db import SessionLocal, TrainingRun
from . import model as cnn
from .dataset import load_training_set


class FaceService:
    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._classes: list[str] = []
        self._loaded = False
        self.status = {"state": "idle", "epoch": 0, "epochs": 0, "message": "", "run_id": None}

    # ---------- model --------------------------------------------------------
    def _ensure_loaded(self):
        if not self._loaded:
            with self._lock:
                if not self._loaded:
                    self._model, self._classes = cnn.load()
                    self._loaded = True

    @property
    def ready(self) -> bool:
        self._ensure_loaded()
        return self._model is not None

    @property
    def classes(self) -> list[str]:
        self._ensure_loaded()
        return list(self._classes)

    def predict(self, face_rgb: np.ndarray, top_k: int = 3) -> list[dict]:
        """Top-k (label, probability) for one 100x100 RGB face."""
        self._ensure_loaded()
        if self._model is None:
            raise RuntimeError("No trained model yet. An admin must upload photos and train.")
        batch = np.expand_dims(face_rgb.astype("float32"), 0)   # 0-255; model rescales itself
        probs = self._model.predict(batch, verbose=0)[0]
        order = np.argsort(probs)[::-1][:top_k]
        return [{"label": self._classes[i], "confidence": round(float(probs[i]), 4)} for i in order]

    @staticmethod
    def decide(top: list[dict]) -> str:
        """Open-set rule: a low-confidence guess is reported as UNKNOWN, not as a wrong person."""
        return "MATCH" if top and top[0]["confidence"] >= settings.CONFIDENCE_THRESHOLD else "UNKNOWN"

    # ---------- training ---------------------------------------------------
    def start_training(self, started_by: str, epochs: int | None = None) -> int:
        if self.status["state"] == "running":
            raise RuntimeError("Training is already running.")
        db = SessionLocal()
        run = TrainingRun(started_by=started_by, status="running")
        db.add(run)
        db.commit()
        run_id = run.id
        db.close()
        self.status = {"state": "running", "epoch": 0, "epochs": epochs or settings.EPOCHS,
                       "message": "Loading face images...", "run_id": run_id}
        threading.Thread(target=self._train_job, args=(run_id, epochs), daemon=True).start()
        return run_id

    def _progress(self, epoch, epochs, logs):
        self.status.update(epoch=epoch, epochs=epochs,
                           message=f"Epoch {epoch}/{epochs} - loss {logs.get('loss', 0):.3f}, "
                                   f"val acc {logs.get('val_accuracy', 0):.2%}")

    def _train_job(self, run_id: int, epochs: int | None):
        db = SessionLocal()
        run = db.get(TrainingRun, run_id)
        try:
            images, labels = load_training_set()
            run.num_images, run.num_classes = len(labels), len(set(labels))
            db.commit()
            metrics = cnn.train(images, labels, epochs=epochs, progress=self._progress)
            with self._lock:
                self._model, self._classes = cnn.load()
                self._loaded = True
            run.status, run.accuracy, run.macro_f1 = "done", metrics["accuracy"], metrics["f1_macro"]
            run.message = (f"Trained on {metrics['num_images']} images of "
                           f"{len(metrics['classes'])} people. Validation accuracy "
                           f"{metrics['accuracy']:.1%}, macro F1 {metrics['f1_macro']:.2f}.")
            if metrics["dropped_classes"]:
                run.message += (f" Skipped (fewer than {settings.MIN_IMAGES_PER_PERSON} images): "
                                f"{', '.join(metrics['dropped_classes'])}.")
            self.status.update(state="done", message=run.message)
        except Exception as exc:          # report any failure to the admin screen
            traceback.print_exc()
            run.status, run.message = "failed", str(exc)
            self.status.update(state="failed", message=str(exc))
        finally:
            run.finished_at = datetime.now(timezone.utc)
            db.commit()
            db.close()


face_service = FaceService()
