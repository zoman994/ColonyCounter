"""Application state — data model without any UI dependency."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class AppState:
    """All mutable application state in one place."""

    image_paths: list[str] = field(default_factory=list)
    image_data: dict[str, dict | None] = field(default_factory=dict)
    display_names: dict[str, str] = field(default_factory=dict)
    manual_marks: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    excluded_auto: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    dish_overrides: dict[str, list] = field(default_factory=dict)
    annotations: dict[str, list[str]] = field(default_factory=dict)
    current_path: str | None = None

    def add_image(self, path: str, name: str | None = None) -> bool:
        if path in self.image_paths:
            return False
        self.image_paths.append(path)
        self.image_data[path] = None
        self.display_names[path] = name or Path(path).name
        return True

    def remove_image(self, path: str) -> None:
        if path in self.image_paths:
            self.image_paths.remove(path)
        for d in (self.image_data, self.display_names, self.manual_marks,
                  self.excluded_auto, self.dish_overrides, self.annotations):
            d.pop(path, None)
        if self.current_path == path:
            self.current_path = None

    def clear(self) -> None:
        for d in (self.image_paths, self.image_data, self.display_names,
                  self.manual_marks, self.excluded_auto, self.dish_overrides,
                  self.annotations):
            if isinstance(d, list):
                d.clear()
            else:
                d.clear()
        self.current_path = None

    @property
    def processed_paths(self) -> list[str]:
        return [p for p in self.image_paths if self.image_data.get(p)]
