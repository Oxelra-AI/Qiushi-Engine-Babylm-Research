"""Run an original training script with isolated scientific output files."""

from __future__ import annotations

import os
from pathlib import Path
import signal
import subprocess


def run_training(command, *, output_dir, timeout, env=None, logf=None):
    """Preserve the supplied trainer arguments, CUDA requirement and time limit."""
    output_dir = Path(output_dir)
    child_env = dict(os.environ if env is None else env)
    child_env["PYTHONUNBUFFERED"] = "1"
    if timeout < 0:
        raise ValueError("Training timeout must be nonnegative")
    if not command:
        raise ValueError("An original trainer command is required")
    if output_dir.exists():
        raise FileExistsError(f"Training output already exists: {output_dir}")

    # Probe in a separate process so the trainer's initialization/RNG is untouched.
    subprocess.run(
        [command[0], "-c", "import torch; raise SystemExit(0 if torch.cuda.is_available() else 2)"],
        env=child_env,
        check=True,
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    stdout_path = output_dir / "stdout.log"
    stderr_path = output_dir / "stderr.log"
    line = "$ " + " ".join(command)
    print(line, flush=True)
    if logf is not None:
        logf.write("\n" + line + "\n")
        logf.flush()

    with stdout_path.open("w", encoding="utf-8") as stdout, stderr_path.open("w", encoding="utf-8") as stderr:
        process = subprocess.Popen(
            command, env=child_env, text=True, stdout=stdout, stderr=stderr,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=timeout or None)
        except BaseException:
            # The original trainers use DataLoader workers; stop the whole group.
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait()
            raise

    stdout_text = stdout_path.read_text(encoding="utf-8")
    stderr_text = stderr_path.read_text(encoding="utf-8")
    print(stdout_text[-5000:], flush=True)
    if stderr_text:
        print(stderr_text[-5000:], flush=True)
    if logf is not None:
        logf.write(stdout_text)
        logf.write(stderr_text)
        logf.write(f"\n[returncode={returncode}]\n")
        logf.flush()
    if returncode:
        raise subprocess.CalledProcessError(returncode, command, stdout_text, stderr_text)
    if not (output_dir / "scientific_metrics.json").is_file():
        raise FileNotFoundError(f"Trainer did not produce scientific_metrics.json: {output_dir}")
    return subprocess.CompletedProcess(command, returncode, stdout_text, stderr_text)
