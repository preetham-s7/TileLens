"""CLI input errors should be actionable, without Python tracebacks."""

import subprocess
import sys
import unittest


class TestCLIValidation(unittest.TestCase):
    def test_invalid_gemm_settings(self):
        for command in ("gemm", "compare", "export-viz"):
            for flag, value in (("-M", "0"), ("-N", "-1"), ("-K", "1.5"),
                                ("--tile-m", "0"), ("--tile-n", "-1"), ("--tile-k", "0")):
                with self.subTest(command=command, flag=flag):
                    self.assert_cli_error(command, flag, value)
        self.assert_cli_error("gemm", "--stages", "-1")

    def assert_cli_error(self, *args):
        result = subprocess.run(
            [sys.executable, "-m", "tilelens.cli.main", *args],
            capture_output=True, text=True,
        )
        self.assertEqual(result.returncode, 2)
        self.assertIn("must be a positive integer", result.stderr)
        self.assertNotIn("Traceback", result.stderr)
