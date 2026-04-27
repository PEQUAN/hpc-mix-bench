from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional
import re


@dataclass
class Task:
    folder: str
    script: str
    index: int
    run_experiments: bool
    run_plotting: bool
    run_debug: bool
    debug_script: Optional[str] = None

    def to_dict(self):
        return asdict(self)


def normalize_bool(x: str) -> bool:
    return str(x).strip().lower() in {"1", "true", "y", "yes"}


def discover_valid_folders(target_folders: List[str]) -> List[Path]:
    folders: List[Path] = []

    if not target_folders:
        for script in Path(".").glob("*/*/run_setting_*.py"):
            d = script.parent
            if (d / "promise.yml").exists():
                folders.append(d)
    else:
        for f in target_folders:
            p = Path(f)
            if p.is_dir():
                folders.append(p)

    unique = sorted(set(x.resolve() for x in folders))
    return unique


def build_tasks(
    target_folders: List[str],
    run_experiments: bool,
    run_plotting: bool,
    run_debug: bool,
) -> List[Task]:
    valid_folders = discover_valid_folders(target_folders)
    tasks: List[Task] = []

    pattern = re.compile(r"run_setting_(\d+)\.py$")

    for folder in valid_folders:
        if not (folder / "promise.yml").exists():
            continue

        scripts = sorted(folder.glob("run_setting_*.py"))
        for script in scripts:
            m = pattern.match(script.name)
            if not m:
                continue

            idx = int(m.group(1))
            debug_script = folder / f"run_debug_{idx}.sh"

            tasks.append(
                Task(
                    folder=str(folder),
                    script=str(script),
                    index=idx,
                    run_experiments=run_experiments,
                    run_plotting=run_plotting,
                    run_debug=run_debug,
                    debug_script=str(debug_script) if debug_script.exists() else None,
                )
            )

    return tasks