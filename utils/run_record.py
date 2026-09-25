"""Record what a run was made with: the notebook, the settings and the environment.

Every notebook calls save_run_record() once it knows where its results go. It
makes a folder named after the notebook and the time, for example
training/03_train_model_20260924120000/, holding a copy of the notebook as it
was run, requirements.txt and run_record.txt. The code itself is not copied
there: the dataset notebook snapshots it once into the project as
utils_YYYYMMDDHHMMSS, and every later notebook runs from that copy.
"""

import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

RECORD_FILE_NAME = "run_record.txt"
REQUIREMENTS_FILE_NAME = "requirements.txt"


def _run(command):
    """Run a command and return its output, or a short note when it is unavailable."""
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=60, check=False
        )
    except (OSError, subprocess.SubprocessError) as error:
        return f"(not available: {type(error).__name__}: {error})"
    if completed.returncode != 0:
        return f"(not available: exit {completed.returncode})"
    return completed.stdout.strip()


def snapshot_utils(
    utils_folder,
    project_folder,
):
    """Copy the utils folder into a project as utils_YYYYMMDDHHMMSS.

    The dataset notebook calls this when it creates a project, and every later
    notebook imports its code from the snapshot rather than from DiverScan/utils.
    A project therefore keeps running with the code it was built with, even
    after the shared utils folder has changed, and the copy travels with the
    project when it is moved. Returns the path of the snapshot.
    """
    utils_folder = Path(utils_folder)
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    snapshot = Path(project_folder) / f"utils_{stamp}"
    snapshot.mkdir(parents=True, exist_ok=False)

    copied = 0
    for path in sorted(utils_folder.glob("*.py")):
        if path.name.startswith("."):
            continue
        shutil.copy2(path, snapshot / path.name)
        copied += 1
    if copied == 0:
        raise FileNotFoundError(f"No .py file found in {utils_folder}; is this the utils folder?")

    print(f"utils snapshot: {snapshot} ({copied} files)")
    return snapshot


def snapshot_notebook(
    notebook_path,
    record_folder,
):
    """Save a copy of the running notebook, or script, into record_folder.

    In Colab the notebook is fetched from the frontend, so the copy holds the
    form values as they are at this moment, whether saved or not. Elsewhere
    the file at notebook_path is copied. Returns the copy's path, or None when
    neither is possible; the run record then says so.
    """
    notebook_path = Path(notebook_path)
    target = Path(record_folder) / notebook_path.name

    if notebook_path.suffix == ".ipynb":
        try:
            from google.colab import _message

            ipynb = _message.blocking_request("get_ipynb", request="", timeout_sec=30)["ipynb"]
            target.write_text(json.dumps(ipynb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
            return target
        except Exception:
            pass  # not in Colab, or the frontend did not answer: fall back to the file

    if notebook_path.exists():
        shutil.copy2(notebook_path, target)
        return target
    return None


def write_requirements(
    save_folder,
):
    """Write the installed packages to requirements.txt."""
    path = Path(save_folder) / REQUIREMENTS_FILE_NAME
    packages = _run([sys.executable, "-m", "pip", "freeze"])
    path.write_text(packages + "\n", encoding="utf-8")
    return path


def save_run_record(
    utils_folder,
    save_folder,
    notebook_path,
    settings=None,
):
    """Record the notebook, the environment and the settings of a run.

    Creates save_folder/<notebook stem>_YYYYMMDDHHMMSS/ holding:
      - a copy of the notebook (or script) at notebook_path, as it was run
      - requirements.txt
      - run_record.txt  timestamp, machine, versions, GPU and the settings

    utils_folder is the code the run imports from, normally the project's
    utils_YYYYMMDDHHMMSS snapshot; it is recorded, not copied. settings is an
    optional dict of the values the run was started with. Returns the folder.
    """
    notebook_path = Path(notebook_path)
    stamp = datetime.now().strftime("%Y%m%d%H%M%S")
    record_folder = Path(save_folder) / f"{notebook_path.stem}_{stamp}"
    record_folder.mkdir(parents=True, exist_ok=False)

    notebook_copy = snapshot_notebook(notebook_path, record_folder)
    write_requirements(record_folder)

    lines = [
        "=== Run ===",
        f"timestamp: {datetime.now(timezone.utc).isoformat(timespec='seconds')}",
        f"host: {platform.node()}",
        f"platform: {platform.platform()}",
        f"python: {sys.version.split()[0]}",
        f"utils_folder: {utils_folder}",
        f"notebook: {notebook_copy.name if notebook_copy else f'(not found: {notebook_path})'}",
        "",
        "=== Versions ===",
    ]
    for module_name in ("numpy", "torch", "pandas", "optuna"):
        try:
            module = __import__(module_name)
            lines.append(f"{module_name}: {getattr(module, '__version__', 'unknown')}")
        except ImportError:
            lines.append(f"{module_name}: (not installed)")
    try:
        import torch

        lines.append(f"cuda: {torch.version.cuda}")
        lines.append(f"cuda available: {torch.cuda.is_available()}")
    except ImportError:
        pass

    lines += ["", "=== GPU ===", _run(["nvidia-smi"])]

    if settings:
        lines += ["", "=== Settings ==="]
        lines += [f"{key}: {value}" for key, value in settings.items()]

    (record_folder / RECORD_FILE_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"Run record written to {record_folder}")
    print(f"  {notebook_copy.name if notebook_copy else '(notebook not found)'}, "
          f"{REQUIREMENTS_FILE_NAME}, {RECORD_FILE_NAME}")

    return record_folder
