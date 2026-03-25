"""Tests that run without GUI, TensorFlow, or legacy binary dependencies."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from ci_validate import validate_project  # noqa: E402


class ProjectIntegrityTests(unittest.TestCase):
    def test_clean_checkout_contract(self) -> None:
        report = validate_project()
        self.assertEqual(report["source"], "Main.py")
        self.assertGreater(report["notebook_cells"], 0)
        self.assertEqual(len(report["datasets"]), 2)


if __name__ == "__main__":
    unittest.main()
