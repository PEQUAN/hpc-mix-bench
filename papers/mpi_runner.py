from __future__ import annotations

import argparse
import os
import stat
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

from mpi4py import MPI
from tasklib import build_tasks, normalize_bool


TAG_READY = 1
TAG_TASK = 2
TAG_DONE = 3
TAG_STOP = 4


def ensure_log_dir(task: Dict[str, Any]) -> Path:
    folder_name = Path(task["folder"]).name
    log_dir = Path("logs") / folder_name
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def run_one_task(task: Dict[str, Any]) -> Dict[str, Any]:
    folder = Path(task["folder"])
    script = Path(task["script"])
    idx = task["index"]
    run_experiments = "true" if task["run_experiments"] else "false"
    run_plotting = "true" if task["run_plotting"] else "false"
    run_debug = bool(task["run_debug"])
    debug_script = task.get("debug_script")

    log_dir = ensure_log_dir(task)
    logfile = log_dir / f"run_{idx}.log"

    env = os.environ.copy()
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    env.setdefault("OPENBLAS_NUM_THREADS", "1")

    start = time.time()

    with logfile.open("a", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write(f"Task: {script.name}\n")
        f.write(f"Folder: {folder}\n")
        f.write(f"Start: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 60 + "\n")

        cmd = [sys.executable, script.name, run_experiments, run_plotting]
        f.write(f"[RUN] {' '.join(cmd)}\n")
        f.flush()

        p = subprocess.run(
            cmd,
            cwd=str(folder),
            stdout=f,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )

        if p.returncode != 0:
            return {
                "ok": False,
                "task": task,
                "stage": "run_setting",
                "returncode": p.returncode,
                "elapsed": time.time() - start,
            }

        if run_debug:
            if debug_script:
                debug_path = Path(debug_script)
                mode = debug_path.stat().st_mode
                debug_path.chmod(mode | stat.S_IXUSR)

                debug_cmd = [f"./{debug_path.name}"]
                f.write(f"[RUN] {' '.join(debug_cmd)}\n")
                f.flush()

                p2 = subprocess.run(
                    debug_cmd,
                    cwd=str(folder),
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    env=env,
                    text=True,
                )
                if p2.returncode != 0:
                    return {
                        "ok": False,
                        "task": task,
                        "stage": "run_debug",
                        "returncode": p2.returncode,
                        "elapsed": time.time() - start,
                    }
            else:
                f.write("[SKIP] debug script missing\n")

    return {
        "ok": True,
        "task": task,
        "elapsed": time.time() - start,
    }


def print_summary_header(size: int, num_tasks: int, mode: str) -> None:
    print("=" * 60)
    print(f"MPI benchmark runner started with {size} ranks")
    print(f"Execution mode: {mode}")
    print(f"Discovered {num_tasks} task(s)")
    print("=" * 60)


def print_summary_footer(completed: int, failed: int, total_elapsed: float) -> int:
    h = int(total_elapsed // 3600)
    m = int((total_elapsed % 3600) // 60)
    s = int(total_elapsed % 60)

    print("=" * 60)
    print("All tasks finished.")
    print(f"Completed : {completed}")
    print(f"Failed    : {failed}")
    print(f"Elapsed   : {h}h {m}m {s}s")
    print("=" * 60)

    return 0 if failed == 0 else 1


def master(comm, size: int, tasks: List[Dict[str, Any]]) -> int:
    num_tasks = len(tasks)

    if size == 1:
        start_time = time.time()
        completed = 0
        failed = 0

        print_summary_header(size=size, num_tasks=num_tasks, mode="serial fallback")

        for task in tasks:
            result = run_one_task(task)
            task_info = result["task"]
            name = Path(task_info["folder"]).name
            script = Path(task_info["script"]).name

            if result["ok"]:
                completed += 1
                print(f"[DONE] rank=0 {name}/{script} ({result['elapsed']:.1f}s)")
            else:
                failed += 1
                print(
                    f"[FAILED] rank=0 {name}/{script} "
                    f"stage={result.get('stage')} rc={result.get('returncode')} "
                    f"({result['elapsed']:.1f}s)"
                )

        total_elapsed = time.time() - start_time
        return print_summary_footer(completed, failed, total_elapsed)

    next_task_idx = 0
    closed_workers = 0
    completed = 0
    failed = 0
    start_time = time.time()

    print_summary_header(size=size, num_tasks=num_tasks, mode="mpi master-worker")

    while closed_workers < size - 1:
        status = MPI.Status()
        msg = comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
        source = status.Get_source()
        tag = status.Get_tag()

        if tag == TAG_READY:
            if next_task_idx < num_tasks:
                comm.send(tasks[next_task_idx], dest=source, tag=TAG_TASK)
                next_task_idx += 1
            else:
                comm.send(None, dest=source, tag=TAG_STOP)

        elif tag == TAG_DONE:
            task = msg["task"]
            name = Path(task["folder"]).name
            script = Path(task["script"]).name

            if msg["ok"]:
                completed += 1
                print(f"[DONE] rank={source} {name}/{script} ({msg['elapsed']:.1f}s)")
            else:
                failed += 1
                print(
                    f"[FAILED] rank={source} {name}/{script} "
                    f"stage={msg.get('stage')} rc={msg.get('returncode')} "
                    f"({msg['elapsed']:.1f}s)"
                )

        elif tag == TAG_STOP:
            closed_workers += 1

    total_elapsed = time.time() - start_time
    return print_summary_footer(completed, failed, total_elapsed)


def worker(comm, rank: int) -> None:
    while True:
        comm.send({"rank": rank}, dest=0, tag=TAG_READY)
        status = MPI.Status()
        task = comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
        tag = status.Get_tag()

        if tag == TAG_STOP:
            comm.send({"rank": rank}, dest=0, tag=TAG_STOP)
            break

        result = run_one_task(task)
        comm.send(result, dest=0, tag=TAG_DONE)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_exp", nargs="?", default="true")
    parser.add_argument("run_plot", nargs="?", default="true")
    parser.add_argument("run_debug", nargs="?", default="false")
    parser.add_argument("folders", nargs="*")
    args = parser.parse_args()

    run_exp = normalize_bool(args.run_exp)
    run_plot = normalize_bool(args.run_plot)
    run_debug = normalize_bool(args.run_debug)

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        tasks = [t.to_dict() for t in build_tasks(args.folders, run_exp, run_plot, run_debug)]
        rc = master(comm, size, tasks)
    else:
        worker(comm, rank)
        rc = 0

    sys.exit(rc)


if __name__ == "__main__":
    main()