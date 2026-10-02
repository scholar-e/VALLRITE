"""Thermal policy and process control tests; never signal a real trainer."""
import logging
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from VisualPhoneme.thermal import ProcessTree, monitor, thermal_action


class ThermalTests(unittest.TestCase):
    def test_thresholds_and_hysteresis(self):
        for gpu, cpu, cooling, expected in [
            (79, 84, False, "run"), (80, 60, False, "cool"),
            (50, 85, False, "cool"), (75, 70, True, "cool"),
            (70, 80, True, "cool"), (74, 79, True, "run"),
            (95, 60, False, "cool"), (50, 95, False, "cool"),
            (96, 60, False, "terminate"), (50, 95.1, True, "terminate"),
            (None, 60, False, "unavailable"), (50, None, True, "unavailable"),
            (96, None, False, "terminate"),
        ]:
            with self.subTest(gpu=gpu, cpu=cpu, cooling=cooling):
                self.assertEqual(thermal_action(gpu, cpu, cooling, True), expected)
        self.assertEqual(thermal_action(None, 60, False, False), "run")

    def test_emergency_kills_and_marks_output(self):
        with tempfile.TemporaryDirectory() as directory, \
                patch("VisualPhoneme.thermal.ProcessTree") as tree_class, \
                patch("VisualPhoneme.thermal.temperatures", return_value=(96, 65)), \
                patch("VisualPhoneme.thermal.signal.signal"):
            tree = tree_class.return_value
            tree.alive.return_value = True
            monitor(12345, Path(directory), True)
            tree.stop.assert_called_once()
            self.assertEqual(tree.signal_stopped.call_args_list[0].args, (signal.SIGKILL,))
            self.assertTrue((Path(directory) / "PAUSE").exists())

    def test_process_tree_stops_resumes_and_kills_child(self):
        with tempfile.TemporaryDirectory() as directory:
            child_file = Path(directory) / "child"
            code = (
                "import subprocess,sys,time; from pathlib import Path; "
                "child=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                "Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)"
            )
            process = subprocess.Popen([sys.executable, "-c", code, str(child_file)])
            tree = ProcessTree(process.pid)
            try:
                deadline = time.monotonic() + 5
                while not child_file.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                child = int(child_file.read_text())
                tree.stop()
                self.assertEqual(set(tree.stopped), {process.pid, child})
                tree.signal_stopped(signal.SIGCONT)
                self.assertFalse(tree.stopped)
                self.assertIsNone(process.poll())
                tree.stop()
                tree.signal_stopped(signal.SIGKILL)
                self.assertEqual(process.wait(timeout=5), -signal.SIGKILL)
            finally:
                tree.stop()
                tree.signal_stopped(signal.SIGKILL)
                process.wait(timeout=5)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    unittest.main()
