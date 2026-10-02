"""Independent Linux thermal watchdog for a trainer and its worker processes."""
from __future__ import annotations

import argparse
import logging
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

LOGGER = logging.getLogger("visual_phoneme.thermal")
CPU_DRIVERS = {"coretemp", "k10temp", "k8temp", "zenpower"}


def temperatures(gpu_required: bool) -> tuple[float | None, float | None]:
    """Read hottest CPU sensor and GPU; missing readings never mean cool."""
    cpu_values = []
    for sensor in Path("/sys/class/hwmon").glob("hwmon*"):
        try:
            if (sensor / "name").read_text().strip() not in CPU_DRIVERS:
                continue
            for source in sensor.glob("temp*_input"):
                cpu_values.append(int(source.read_text().strip()) / 1000)
        except (OSError, ValueError):
            continue
    gpu = None
    if gpu_required:
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, check=True, timeout=2,
            )
            values = [float(value) for value in result.stdout.splitlines()]
            if values and all(math.isfinite(value) and value >= 0 for value in values):
                gpu = max(values)
        except (OSError, ValueError, subprocess.SubprocessError):
            pass
    return gpu, max(cpu_values) if cpu_values else None


def thermal_action(gpu, cpu, cooling: bool, gpu_required: bool) -> str:
    if any(value is not None and value > 95 for value in (gpu, cpu)):
        return "terminate"
    if cpu is None or (gpu_required and gpu is None):
        return "unavailable"
    if cpu >= 85 or (gpu is not None and gpu >= 80):
        return "cool"
    if cooling and (cpu >= 80 or (gpu is not None and gpu >= 75)):
        return "cool"
    return "run"


def process_info(pid: int) -> tuple[int, str]:
    fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
    return int(fields[1]), fields[19]  # parent PID and kernel start time


class ProcessTree:
    def __init__(self, parent: int):
        self.parent = parent
        self.identity = process_info(parent)[1]
        self.stopped: dict[int, str] = {}

    def alive(self) -> bool:
        try:
            return process_info(self.parent)[1] == self.identity
        except (OSError, ValueError, IndexError):
            return False

    def stop(self, pid=None):
        pid = self.parent if pid is None else pid
        try:
            identity = process_info(pid)[1]
            if pid == self.parent and identity != self.identity:
                return
            os.kill(pid, signal.SIGSTOP)
            self.stopped[pid] = identity
        except (OSError, ValueError, IndexError):
            return
        # Stop the parent first so it cannot create more workers during traversal.
        for entry in Path("/proc").iterdir():
            if not entry.name.isdigit() or int(entry.name) == os.getpid():
                continue
            try:
                if process_info(int(entry.name))[0] == pid:
                    self.stop(int(entry.name))
            except (OSError, ValueError, IndexError):
                continue

    def signal_stopped(self, signum):
        # Resume/kill children first and the trainer last.
        for pid, identity in reversed(list(self.stopped.items())):
            try:
                if process_info(pid)[1] == identity:
                    os.kill(pid, signum)
            except (OSError, ValueError, IndexError):
                continue
        self.stopped.clear()


class ThermalGuard:
    """Lifetime follows the training main function, including exceptional exits."""
    def __init__(self, output_dir: Path, gpu_required: bool):
        # Execute the standalone module file to avoid importing torch in the watchdog.
        self.command = [sys.executable, str(Path(__file__).resolve()),
                        "--parent", str(os.getpid()), "--output-dir", str(output_dir)]
        if gpu_required:
            self.command.append("--gpu-required")

    def __enter__(self):
        self.process = subprocess.Popen(self.command, start_new_session=True)
        return self

    def __exit__(self, *_):
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait(timeout=10)


def monitor(parent: int, output_dir: Path, gpu_required: bool):
    tree = ProcessTree(parent)
    running = True

    def shutdown(_signum, _frame):
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    LOGGER.info("Thermal protection: GPU 80 °C, CPU 85 °C; resume below GPU 75 °C "
                "and CPU 80 °C; terminate above 95 °C; polling every 1s")
    previous = "run"
    try:
        while running and tree.alive():
            gpu, cpu = temperatures(gpu_required)
            action = thermal_action(gpu, cpu, bool(tree.stopped), gpu_required)
            if action == "terminate":
                tree.stop()
                LOGGER.critical("THERMAL EMERGENCY: GPU=%s °C CPU=%s °C; "
                                "killing training and workers without checkpointing", gpu, cpu)
                try:
                    (output_dir / "PAUSE").touch()
                finally:
                    tree.signal_stopped(signal.SIGKILL)
                return
            if action in {"cool", "unavailable"}:
                tree.stop()
                if action != previous:
                    LOGGER.warning("Thermal pause (%s): GPU=%s °C CPU=%s °C", action, gpu, cpu)
            else:
                if tree.stopped:
                    LOGGER.info("Thermal resume: GPU=%s °C CPU=%s °C", gpu, cpu)
                    tree.signal_stopped(signal.SIGCONT)
            previous = action
            time.sleep(1)
    except Exception:
        # A broken watchdog must not leave unprotected training running.
        LOGGER.exception("Thermal watchdog failed; terminating training")
        tree.stop()
        tree.signal_stopped(signal.SIGKILL)
        raise
    finally:
        tree.signal_stopped(signal.SIGCONT if tree.alive() else signal.SIGKILL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--gpu-required", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        handlers=[logging.FileHandler(args.output_dir / "train.log"), logging.StreamHandler()],
    )
    monitor(args.parent, args.output_dir, args.gpu_required)


if __name__ == "__main__":
    main()
