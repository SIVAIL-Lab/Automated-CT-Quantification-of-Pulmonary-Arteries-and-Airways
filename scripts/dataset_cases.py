"""Discover cases by their input layout, independently of dataset naming."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path


def _discover(
    root: Path,
    selected: list[str] | None,
    limit: int | None,
    matches: Callable[[Path], bool],
    description: str,
) -> list[Path]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {root}")
    if limit is not None and limit < 1:
        raise ValueError("Case limit must be a positive integer")
    directories = sorted(path for path in root.iterdir() if path.is_dir())
    if selected:
        by_name = {path.name: path for path in directories}
        missing = [name for name in selected if name not in by_name]
        if missing:
            raise FileNotFoundError(f"Unknown case(s): {', '.join(missing)}")
        # Keep explicitly selected incomplete cases so the caller can report the
        # missing input for that case, instead of silently dropping its work.
        cases = [by_name[name] for name in dict.fromkeys(selected)]
    else:
        cases = [path for path in directories if matches(path)]
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise FileNotFoundError(f"No case directories containing {description} found below {root}")
    return cases


def discover_ct_cases(
    root: Path, selected: list[str] | None = None, limit: int | None = None,
) -> list[Path]:
    """Find direct child cases with image/; explicit selections may be incomplete."""
    return _discover(root, selected, limit, lambda case: (case / "image").is_dir(), "image/")


def discover_mask_cases(
    root: Path, selected: list[str] | None = None, limit: int | None = None,
    *, mask_name: str,
) -> list[Path]:
    """Find direct child cases with the mask required by the consuming stage."""
    return _discover(root, selected, limit, lambda case: (case / mask_name).is_file(), mask_name)
