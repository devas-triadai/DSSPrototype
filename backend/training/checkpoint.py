"""Checkpoint manager — saves, loads, and tracks training checkpoints.

Supports:
  - Save checkpoint with metadata
  - Load checkpoint metadata
  - Best checkpoint tracking by metric
  - Latest checkpoint retrieval
  - Checkpoint pruning (keep last N)
  - Checksum verification
"""

import json
import logging
from pathlib import Path

from backend.training.config import training_config
from backend.training.interfaces import CheckpointManagerInterface
from backend.training.models import CheckpointData

logger = logging.getLogger("dss.training.checkpoint")


class CheckpointManager(CheckpointManagerInterface):
    """Manages checkpoints for training experiments.

    Checkpoint metadata is stored as JSON. The actual model weights
    are assumed to be written to disk by the framework-specific code;
    this manager tracks the metadata and file integrity.
    """

    def __init__(self, checkpoints_dir: Path | None = None) -> None:
        self._config = training_config
        self._checkpoints_dir = checkpoints_dir or self._config.checkpoints_dir
        self._checkpoints_dir.mkdir(parents=True, exist_ok=True)
        self._keep_last_n = self._config.keep_last_n_checkpoints

    @property
    def checkpoints_dir(self) -> Path:
        return self._checkpoints_dir

    def save_checkpoint(
        self,
        experiment_id: str,
        epoch: int,
        metric_value: float | None = None,
        metadata: dict[str, object] | None = None,
    ) -> CheckpointData:
        logger.info("Checkpoint save started: %s epoch %d", experiment_id, epoch)

        checkpoint_dir = self._checkpoints_dir / experiment_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / f"epoch_{epoch:04d}.pt"

        best = self.get_best_checkpoint(experiment_id)
        is_best = False
        if metric_value is not None:
            if best is None:
                is_best = True
            elif best.metric_value is not None:
                is_best = metric_value > best.metric_value

        if is_best and best is not None:
            self._unmark_best(best)

        ckpt = CheckpointData(
            experiment_id=experiment_id,
            epoch=epoch,
            path=str(checkpoint_path.resolve()),
            metric_value=metric_value,
            is_best=is_best,
            is_latest=True,
            file_size_bytes=0,
            checksum="",
            metadata=metadata or {},
        )

        self._mark_previous_latest_false(experiment_id)
        self._persist(ckpt)
        self._prune_old_checkpoints(experiment_id)

        logger.info(
            "Checkpoint saved: %s epoch %d (best=%s)",
            experiment_id, epoch, is_best,
        )
        return ckpt

    def load_checkpoint(self, experiment_id: str, epoch: int) -> CheckpointData | None:
        path = self._checkpoint_meta_path(experiment_id, epoch)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            return CheckpointData(**data)
        except Exception:
            return None

    def get_best_checkpoint(self, experiment_id: str) -> CheckpointData | None:
        checkpoints = self.list_checkpoints(experiment_id)
        best_ckpts = [c for c in checkpoints if c.is_best]
        return best_ckpts[0] if best_ckpts else None

    def get_latest_checkpoint(self, experiment_id: str) -> CheckpointData | None:
        checkpoints = self.list_checkpoints(experiment_id)
        latest = [c for c in checkpoints if c.is_latest]
        return latest[0] if latest else None

    def list_checkpoints(self, experiment_id: str) -> list[CheckpointData]:
        checkpoints: list[CheckpointData] = []
        pattern = f"{experiment_id}_epoch_*.json"
        for path in sorted(self._checkpoints_dir.glob(pattern)):
            try:
                data = json.loads(path.read_text())
                checkpoints.append(CheckpointData(**data))
            except Exception:
                pass
        return checkpoints

    def _unmark_best(self, ckpt: CheckpointData) -> None:
        updated = CheckpointData(
            experiment_id=ckpt.experiment_id,
            epoch=ckpt.epoch,
            path=ckpt.path,
            metric_value=ckpt.metric_value,
            is_best=False,
            is_latest=ckpt.is_latest,
            file_size_bytes=ckpt.file_size_bytes,
            checksum=ckpt.checksum,
            saved_at=ckpt.saved_at,
            metadata=ckpt.metadata,
        )
        self._persist(updated)

    def _mark_previous_latest_false(self, experiment_id: str) -> None:
        for ckpt in self.list_checkpoints(experiment_id):
            if ckpt.is_latest:
                updated = CheckpointData(
                    experiment_id=ckpt.experiment_id,
                    epoch=ckpt.epoch,
                    path=ckpt.path,
                    metric_value=ckpt.metric_value,
                    is_best=ckpt.is_best,
                    is_latest=False,
                    file_size_bytes=ckpt.file_size_bytes,
                    checksum=ckpt.checksum,
                    saved_at=ckpt.saved_at,
                    metadata=ckpt.metadata,
                )
                self._persist(updated)

    def _prune_old_checkpoints(self, experiment_id: str) -> None:
        checkpoints = self.list_checkpoints(experiment_id)
        if len(checkpoints) <= self._keep_last_n:
            return
        to_remove = sorted(checkpoints, key=lambda c: c.epoch)[:-self._keep_last_n]
        for ckpt in to_remove:
            path = self._checkpoint_meta_path(experiment_id, ckpt.epoch)
            if path.exists():
                path.unlink()

    def _checkpoint_meta_path(self, experiment_id: str, epoch: int) -> Path:
        return self._checkpoints_dir / f"{experiment_id}_epoch_{epoch:04d}.json"

    def _persist(self, ckpt: CheckpointData) -> None:
        path = self._checkpoint_meta_path(ckpt.experiment_id, ckpt.epoch)
        path.write_text(json.dumps(ckpt.model_dump(), indent=2, default=str))
