"""Hyperparameter manager — manages named training profiles.

Supports saving, retrieving, listing, deleting, and applying
named hyperparameter profiles with optional overrides.
"""

import json
import logging
from pathlib import Path

from backend.training.config import training_config
from backend.training.interfaces import HyperparameterManagerInterface
from backend.training.models import HyperparameterProfile, TrainingConfigData

logger = logging.getLogger("dss.training.hyperparameter_manager")


_DEFAULT_PROFILES: dict[str, dict[str, object]] = {
    "fast": {
        "name": "fast",
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "optimizer": "adam",
        "scheduler": "cosine",
        "image_size": (416, 416),
        "weight_decay": 0.0005,
        "seed": 42,
        "description": "Quick training for prototyping",
    },
    "balanced": {
        "name": "balanced",
        "learning_rate": 0.001,
        "batch_size": 16,
        "epochs": 100,
        "optimizer": "adam",
        "scheduler": "cosine",
        "image_size": (640, 640),
        "weight_decay": 0.0001,
        "seed": 42,
        "early_stopping_patience": 10,
        "description": "Balanced training for production models",
    },
    "accurate": {
        "name": "accurate",
        "learning_rate": 0.0005,
        "batch_size": 8,
        "epochs": 300,
        "optimizer": "adamw",
        "scheduler": "cosine",
        "image_size": (768, 768),
        "weight_decay": 0.05,
        "seed": 42,
        "early_stopping_patience": 20,
        "early_stopping_delta": 0.0005,
        "mixed_precision": True,
        "warmup_epochs": 3,
        "description": "High-accuracy training with warmup and large image size",
    },
    "tiny": {
        "name": "tiny",
        "learning_rate": 0.002,
        "batch_size": 64,
        "epochs": 50,
        "optimizer": "sgd",
        "scheduler": "step",
        "image_size": (320, 320),
        "weight_decay": 0.001,
        "seed": 42,
        "description": "Tiny model training for edge devices",
    },
}


class HyperparameterManager(HyperparameterManagerInterface):
    """Manages named hyperparameter profiles with JSON persistence.

    Comes pre-loaded with four built-in profiles: fast, balanced,
    accurate, and tiny. Custom profiles can be added and persisted.
    """

    def __init__(self, configs_dir: Path | None = None) -> None:
        self._config = training_config
        self._configs_dir = configs_dir or self._config.configs_dir
        self._configs_dir.mkdir(parents=True, exist_ok=True)
        self._cache: dict[str, HyperparameterProfile] = {}
        self._load_defaults()

    def save_profile(self, profile: HyperparameterProfile) -> HyperparameterProfile:
        logger.info("Saving profile: %s", profile.name)
        self._cache[profile.name] = profile
        self._persist(profile)
        return profile

    def get_profile(self, name: str) -> HyperparameterProfile | None:
        if name in self._cache:
            return self._cache[name]
        path = self._profile_path(name)
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
            profile = HyperparameterProfile(**data)
            self._cache[name] = profile
            return profile
        except Exception:
            return None

    def list_profiles(self) -> list[HyperparameterProfile]:
        profiles: list[HyperparameterProfile] = list(self._cache.values())
        seen = set()
        unique = []
        for p in profiles:
            if p.name not in seen:
                seen.add(p.name)
                unique.append(p)
        for path in sorted(self._configs_dir.glob("*.json")):
            try:
                data = json.loads(path.read_text())
                name = data.get("name", path.stem)
                if name not in seen:
                    profile = HyperparameterProfile(**data)
                    seen.add(name)
                    unique.append(profile)
            except Exception:
                pass
        return unique

    def delete_profile(self, name: str) -> bool:
        if name in _DEFAULT_PROFILES:
            logger.warning("Cannot delete built-in profile: %s", name)
            return False
        self._cache.pop(name, None)
        path = self._profile_path(name)
        if path.exists():
            path.unlink()
            logger.info("Profile deleted: %s", name)
            return True
        return False

    def apply_profile(
        self, profile: HyperparameterProfile, overrides: dict[str, object] | None = None,
    ) -> TrainingConfigData:
        logger.info("Applying profile: %s with overrides: %s", profile.name, overrides or {})

        image_size = profile.image_size
        if overrides and "image_size" in overrides:
            val = overrides["image_size"]
            if isinstance(val, (list, tuple)) and len(val) == 2:
                image_size = (int(val[0]), int(val[1]))

        over = overrides or {}
        config = TrainingConfigData(
            model_name="",
            model_version="1.0.0",
            experiment_name=profile.name,
            batch_size=int(over.get("batch_size", profile.batch_size)),  # type: ignore[call-overload]
            epochs=int(over.get("epochs", profile.epochs)),  # type: ignore[call-overload]
            learning_rate=float(over.get("learning_rate", profile.learning_rate)),  # type: ignore[arg-type]
            optimizer=str(over.get("optimizer", profile.optimizer)),
            scheduler=str(over.get("scheduler", profile.scheduler)),
            weight_decay=float(over.get("weight_decay", profile.weight_decay)),  # type: ignore[arg-type]
            image_size=image_size,
            seed=int(over.get("seed", profile.seed)),  # type: ignore[call-overload]
            early_stopping_patience=over.get(  # type: ignore[arg-type]
                "early_stopping_patience", profile.early_stopping_patience,
            ),
            early_stopping_delta=float(
                over.get("early_stopping_delta", profile.early_stopping_delta),  # type: ignore[arg-type]
            ),
            mixed_precision=bool(over.get("mixed_precision", profile.mixed_precision)),
        )
        return config

    def _load_defaults(self) -> None:
        for name, data in _DEFAULT_PROFILES.items():
            self._cache[name] = HyperparameterProfile(**data)  # type: ignore[arg-type]

    def _profile_path(self, name: str) -> Path:
        return self._configs_dir / f"{name}.json"

    def _persist(self, profile: HyperparameterProfile) -> None:
        path = self._profile_path(profile.name)
        path.write_text(json.dumps(profile.model_dump(), indent=2, default=str))
