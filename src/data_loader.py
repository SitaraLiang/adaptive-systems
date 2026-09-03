"""Loading and validation utilities for the Server Machine Dataset (SMD)."""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

try:
    import numpy as np
except ModuleNotFoundError as exc: 
    raise ModuleNotFoundError(
        "NumPy is required to load SMD. Install the project dependencies first."
    ) from exc

if TYPE_CHECKING:
    from numpy.typing import NDArray


EXPECTED_FEATURES = 38
DEFAULT_SMD_ROOT = Path("data/raw/smd")
MACHINE_NAME_PATTERN = re.compile(r"machine-\d+-\d+")
INTERPRETATION_PATTERN = re.compile(r"(\d+)-(\d+):([\d,]+)")


@dataclass(frozen=True)
class AnomalyInterval:
    """An inclusive, zero-based anomaly interval and its one-based dimensions."""

    start: int
    end: int
    dimensions: tuple[int, ...]


@dataclass(frozen=True)
class SMDMachineData:
    """Train/test arrays and test annotations for one SMD machine."""

    machine: str
    train: NDArray[np.float32]
    test: NDArray[np.float32]
    test_labels: NDArray[np.int8]
    interpretation: tuple[AnomalyInterval, ...]


def _machine_file(root: Path, category: str, machine: str) -> Path:
    path = root / category / f"{machine}.txt"
    if not path.is_file():
        raise FileNotFoundError(f"Missing SMD {category} file: {path}")
    return path


def _machine_sort_key(machine: str) -> tuple[int, int]:
    """Sort machine names numerically, so machine-3-2 precedes machine-3-10."""

    _, group, index = machine.split("-")
    return int(group), int(index)


def discover_smd_machines(
    root: str | Path = DEFAULT_SMD_ROOT,
) -> tuple[str, ...]:
    """Discover available SMD machines from filenames in the train directory."""

    train_directory = Path(root) / "train"
    if not train_directory.is_dir():
        raise FileNotFoundError(f"Missing SMD train directory: {train_directory}")

    machines = {
        path.stem
        for path in train_directory.glob("machine-*.txt")
        if MACHINE_NAME_PATTERN.fullmatch(path.stem)
    }
    if not machines:
        raise FileNotFoundError(f"No SMD machine files found in {train_directory}")

    return tuple(sorted(machines, key=_machine_sort_key))


def load_interpretation_labels(path: Path) -> tuple[AnomalyInterval, ...]:
    """Parse SMD interpretation labels.

    Interval positions are zero-based and inclusive. Dimension identifiers are
    kept one-based, matching the dataset files.
    """

    intervals: list[AnomalyInterval] = []
    with path.open(encoding="utf-8") as file:
        for line_number, raw_line in enumerate(file, start=1):
            line = raw_line.strip()
            if not line:
                continue

            match = INTERPRETATION_PATTERN.fullmatch(line)
            if match is None:
                raise ValueError(
                    f"Invalid interpretation label at {path}:{line_number}: {line!r}"
                )

            start, end = (int(value) for value in match.group(1, 2))
            dimensions = tuple(int(value) for value in match.group(3).split(","))
            if start > end:
                raise ValueError(f"Invalid interval at {path}:{line_number}: {line!r}")
            if any(dimension < 1 or dimension > EXPECTED_FEATURES for dimension in dimensions):
                raise ValueError(
                    f"Dimension outside 1..{EXPECTED_FEATURES} at "
                    f"{path}:{line_number}: {line!r}"
                )

            intervals.append(AnomalyInterval(start, end, dimensions))

    return tuple(intervals)


def load_smd_machine(
    machine: str = "machine-1-1",
    root: str | Path = DEFAULT_SMD_ROOT,
) -> SMDMachineData:
    """Load and validate the four files associated with one SMD machine."""

    if MACHINE_NAME_PATTERN.fullmatch(machine) is None:
        raise ValueError(
            f"Invalid machine name {machine!r}; expected a name like 'machine-1-1'."
        )

    root = Path(root)
    train_path = _machine_file(root, "train", machine)
    test_path = _machine_file(root, "test", machine)
    label_path = _machine_file(root, "test_label", machine)
    interpretation_path = _machine_file(root, "interpretation_label", machine)

    train = np.loadtxt(train_path, delimiter=",", dtype=np.float32)
    test = np.loadtxt(test_path, delimiter=",", dtype=np.float32)
    test_labels = np.loadtxt(label_path, dtype=np.int8)
    interpretation = load_interpretation_labels(interpretation_path)

    for name, values in (("train", train), ("test", test)):
        if values.ndim != 2 or values.shape[1] != EXPECTED_FEATURES:
            raise ValueError(
                f"{name} must have shape (n, {EXPECTED_FEATURES}); got {values.shape}."
            )
        if not np.isfinite(values).all():
            raise ValueError(f"{name} contains NaN or infinite values.")

    if test_labels.ndim != 1 or len(test_labels) != len(test):
        raise ValueError(
            "Test labels must be one-dimensional and match the test length; "
            f"got labels {test_labels.shape} and test {test.shape}."
        )
    if not np.isin(test_labels, (0, 1)).all():
        raise ValueError("Test labels must contain only 0 and 1.")

    for interval in interpretation:
        if interval.end >= len(test):
            raise ValueError(
                f"Interpretation interval {interval.start}-{interval.end} exceeds "
                f"the test length {len(test)}."
            )
        # SMD's two annotation sources do not always use identical boundaries.
        # For example, several machine-1-1 interpretation intervals extend one
        # point beyond the corresponding run in test_label. Require meaningful
        # overlap without treating those source annotations as malformed.
        if not test_labels[interval.start : interval.end + 1].any():
            raise ValueError(
                f"Interpretation interval {interval.start}-{interval.end} does not "
                "overlap any anomaly in test_label."
            )

    return SMDMachineData(machine, train, test, test_labels, interpretation)


def load_all_smd_machines(
    root: str | Path = DEFAULT_SMD_ROOT,
) -> dict[str, SMDMachineData]:
    """Discover, load, and validate every available SMD machine separately."""

    root = Path(root)
    return {
        machine: load_smd_machine(machine, root)
        for machine in discover_smd_machines(root)
    }


def _print_summary(data: SMDMachineData) -> None:
    print(f"machine: {data.machine}")
    print(f"  train: {data.train.shape}")
    print(f"  test: {data.test.shape}")
    print(f"  test anomalies: {int(data.test_labels.sum())}")
    print(f"  interpretation intervals: {len(data.interpretation)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Load and validate SMD machines.")
    parser.add_argument(
        "machine",
        nargs="?",
        help="Machine to load, such as machine-1-1. Omit to load all machines.",
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_SMD_ROOT)
    args = parser.parse_args()

    if args.machine:
        _print_summary(load_smd_machine(args.machine, args.root))
        return

    machines = load_all_smd_machines(args.root)
    print(f"loaded machines: {len(machines)}")
    for data in machines.values():
        _print_summary(data)


if __name__ == "__main__":
    main()
