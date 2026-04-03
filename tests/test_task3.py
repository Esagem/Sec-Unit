"""
Test cases for Task 3: Executor
=================================
One test case per function (4 total), as required by the project spec.
Tests are designed to run without Kubescape installed.
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task3.executor import (
    load_diff_files,
    determine_controls,
    run_kubescape,
    generate_csv,
)


def _write_file(content, directory, filename):
    """Helper to write a text file for testing."""
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        f.write(content)
    return path


class TestLoadDiffFiles(unittest.TestCase):
    """Test 1: load_diff_files"""

    def test_loads_two_valid_diff_files(self):
        """Loads two valid TEXT diff files and returns their content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            names_path = _write_file("Kubelet\nPod Security\n", tmpdir, "names.txt")
            reqs_path = _write_file(
                "Kubelet,Ensure Anonymous Auth is Not Enabled\n", tmpdir, "reqs.txt"
            )

            names_content, reqs_content = load_diff_files(names_path, reqs_path)

            self.assertIn("Kubelet", names_content)
            self.assertIn("Pod Security", names_content)
            self.assertIn("Kubelet,Ensure Anonymous Auth", reqs_content)

    def test_raises_on_missing_file(self):
        """Raises FileNotFoundError for a nonexistent path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            names_path = _write_file("test\n", tmpdir, "names.txt")
            with self.assertRaises(FileNotFoundError):
                load_diff_files(names_path, "/nonexistent/reqs.txt")

    def test_raises_on_empty_file(self):
        """Raises ValueError for an empty file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            names_path = _write_file("test\n", tmpdir, "names.txt")
            empty_path = _write_file("", tmpdir, "empty.txt")
            with self.assertRaises(ValueError):
                load_diff_files(names_path, empty_path)


class TestDetermineControls(unittest.TestCase):
    """Test 2: determine_controls"""

    def test_maps_differences_to_controls(self):
        """Maps element differences to Kubescape control IDs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "controls.txt")

            names_content = "Kubelet\nLogging\n"
            reqs_content = "Kubelet,Ensure Anonymous Auth is Not Enabled\n"

            result_path = determine_controls(
                names_content, reqs_content, output_path=out_path
            )

            self.assertTrue(os.path.exists(result_path))
            content = open(result_path).read()

            # Should contain kubescape control IDs
            self.assertRegex(content, r"C-\d{4}")
            self.assertNotIn("NO DIFFERENCES FOUND", content)

    def test_reports_no_differences(self):
        """Reports no differences when both diffs indicate no changes."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "controls.txt")

            result_path = determine_controls(
                "NO DIFFERENCES IN REGARDS TO ELEMENT NAMES",
                "NO DIFFERENCES IN REGARDS TO ELEMENT REQUIREMENTS",
                output_path=out_path,
            )

            content = open(result_path).read()
            self.assertIn("NO DIFFERENCES FOUND", content)


class TestRunKubescape(unittest.TestCase):
    """Test 3: run_kubescape"""

    @patch("task3.executor.subprocess.run")
    def test_runs_kubescape_with_controls(self, mock_run):
        """Mocks Kubescape execution and verifies DataFrame output."""
        mock_results = {
            "resources": [
                {
                    "resourceID": "res-001",
                    "source": {"relativePath": "deployment.yaml"},
                }
            ],
            "results": [
                {
                    "resourceID": "res-001",
                    "controls": [
                        {
                            "controlID": "C-0057",
                            "name": "Privileged container",
                            "severity": {"scoreFactor": 8},
                            "status": {"status": "failed"},
                        },
                        {
                            "controlID": "C-0057",
                            "name": "Privileged container",
                            "severity": {"scoreFactor": 8},
                            "status": {"status": "passed"},
                        },
                    ],
                }
            ],
        }

        mock_run.return_value = MagicMock(returncode=0, stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            controls_path = _write_file("C-0057\n", tmpdir, "controls.txt")

            # Simulate kubescape writing to CWD (relative path)
            results_json_tmp = "kubescape_results.json"
            with open(results_json_tmp, "w") as f:
                json.dump(mock_results, f)

            try:
                # Create a dummy scan target directory
                scan_dir = os.path.join(tmpdir, "yamls")
                os.makedirs(scan_dir, exist_ok=True)

                df = run_kubescape(
                    controls_path, yamls_path=scan_dir, kubescape_cmd="kubescape"
                )

                self.assertIsInstance(df, pd.DataFrame)
                self.assertIn("Control name", df.columns)
                self.assertIn("Failed resources", df.columns)
                self.assertIn("All Resources", df.columns)
                self.assertIn("Compliance score", df.columns)
                self.assertGreater(len(df), 0)
                self.assertEqual(df["FilePath"].iloc[0], "deployment.yaml")
            finally:
                if os.path.exists(results_json_tmp):
                    os.unlink(results_json_tmp)


class TestRunKubescapeFilePath(unittest.TestCase):
    """Test that FilePath is populated from resources[] array, not rawResource."""

    @patch("task3.executor.subprocess.run")
    def test_filepath_populated_from_resources_array(self, mock_run):
        """FilePath column must come from data['resources'][*].source.relativePath."""
        mock_results = {
            "resources": [
                {
                    "resourceID": "abc-123",
                    "source": {"relativePath": "namespace/pod.yaml"},
                }
            ],
            "results": [
                {
                    "resourceID": "abc-123",
                    "controls": [
                        {
                            "controlID": "C-0044",
                            "name": "Container securityContext",
                            "severity": {"scoreFactor": 5},
                            "status": {"status": "failed"},
                        }
                    ],
                }
            ],
        }

        mock_run.return_value = MagicMock(returncode=0, stderr="")

        with tempfile.TemporaryDirectory() as tmpdir:
            controls_path = _write_file("C-0044\n", tmpdir, "controls.txt")

            # Simulate kubescape writing to CWD (relative path)
            results_json_tmp = "kubescape_results.json"
            with open(results_json_tmp, "w") as f:
                json.dump(mock_results, f)

            try:
                scan_dir = os.path.join(tmpdir, "yamls")
                os.makedirs(scan_dir, exist_ok=True)

                df = run_kubescape(
                    controls_path, yamls_path=scan_dir, kubescape_cmd="kubescape"
                )

                self.assertIsInstance(df, pd.DataFrame)
                self.assertIn("FilePath", df.columns)
                self.assertTrue(
                    (df["FilePath"] == "namespace/pod.yaml").any(),
                    "FilePath must be sourced from resources[] array",
                )
            finally:
                if os.path.exists(results_json_tmp):
                    os.unlink(results_json_tmp)


class TestGenerateCSV(unittest.TestCase):
    """Test 4: generate_csv"""

    def test_generates_csv_with_required_headers(self):
        """Generates a CSV file with the correct headers and data."""
        df = pd.DataFrame(
            [
                {
                    "FilePath": "deployment.yaml",
                    "Severity": 8,
                    "Control name": "C-0057 - Privileged container",
                    "Failed resources": 1,
                    "All Resources": 2,
                    "Compliance score": 50.0,
                },
                {
                    "FilePath": "service.yaml",
                    "Severity": 5,
                    "Control name": "C-0034 - Automatic mapping of SA",
                    "Failed resources": 0,
                    "All Resources": 1,
                    "Compliance score": 100.0,
                },
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "results.csv")
            result_path = generate_csv(df, output_path=out_path)

            self.assertTrue(os.path.exists(result_path))

            result_df = pd.read_csv(result_path)
            expected_cols = [
                "FilePath",
                "Severity",
                "Control name",
                "Failed resources",
                "All Resources",
                "Compliance score",
            ]
            for col in expected_cols:
                self.assertIn(col, result_df.columns)

            self.assertEqual(len(result_df), 2)
            self.assertEqual(result_df.iloc[0]["Compliance score"], 50.0)


if __name__ == "__main__":
    unittest.main()
