from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


class TargetMapError(RuntimeError):
    pass


def normalize_key(label: str) -> str:
    raw = label.strip().lower()
    if raw in {"space", "spacebar", "space bar"}:
        return " "
    if len(raw) != 1:
        raise TargetMapError(f"Expected one key label, got {label!r}")
    return raw


@dataclass(frozen=True)
class ImageKeyTarget:
    label: str
    center_px: tuple[float, float]
    confidence: float
    source: str


@dataclass(frozen=True)
class KeyboardTargetMap:
    path: Path
    raw: dict

    @property
    def accepted(self) -> bool:
        return bool(self.raw.get("calibration", {}).get("accepted", False))

    @property
    def reason(self) -> str:
        return str(self.raw.get("calibration", {}).get("reason", "missing calibration reason"))

    def target_for(self, label: str) -> ImageKeyTarget:
        key = normalize_key(label)
        targets = self.raw.get("key_targets", {})
        if key not in targets:
            raise TargetMapError(f"Target {label!r} is unavailable in {self.path}")
        target = targets[key]
        center = target.get("center_px")
        if not isinstance(center, list) or len(center) != 2:
            raise TargetMapError(f"Target {label!r} has invalid center_px in {self.path}")
        return ImageKeyTarget(
            label=key,
            center_px=(float(center[0]), float(center[1])),
            confidence=float(target.get("confidence", 0.0)),
            source=str(target.get("source", "unknown")),
        )


def load_target_map(path: str | Path, required_key: str | None = None) -> KeyboardTargetMap:
    target_path = Path(path)
    raw = json.loads(target_path.read_text(encoding="utf-8"))
    if raw.get("schema_version") != "0.1":
        raise TargetMapError(f"Unsupported target-map schema: {raw.get('schema_version')!r}")
    target_map = KeyboardTargetMap(path=target_path, raw=raw)
    if not target_map.accepted:
        raise TargetMapError(f"Calibration rejected: {target_map.reason}")
    if required_key is not None:
        target_map.target_for(required_key)
    return target_map

