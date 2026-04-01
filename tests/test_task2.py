"""
Test cases for Task 2: Comparator
===================================
One test case per function (3 total), as required by the project spec.
"""

import os
import sys
import yaml
import tempfile
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task2.comparator import (
    load_yaml_files,
    compare_element_names,
    compare_element_requirements,
)


# Sample KDE data matching Task 1 output schema
SAMPLE_KDES_A = {
    "element1": {
        "name": "Logging",
        "requirements": ["Enable audit Logs", "Ensure audit logs are stored"],
    },
    "element2": {
        "name": "Kubelet",
        "requirements": ["Ensure Anonymous Auth is Not Enabled"],
    },
}

SAMPLE_KDES_B = {
    "element1": {
        "name": "Logging",
        "requirements": ["Enable audit Logs"],
    },
    "element2": {
        "name": "Pod Security",
        "requirements": ["Enforce restricted pod security standards"],
    },
}

SAMPLE_KDES_IDENTICAL = {
    "element1": {
        "name": "Logging",
        "requirements": ["Enable audit Logs"],
    },
}


def _write_yaml(data, directory, filename):
    """Helper to write a YAML file for testing."""
    path = os.path.join(directory, filename)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    return path


class TestLoadYamlFiles(unittest.TestCase):
    """Test 1: load_yaml_files"""

    def test_loads_two_valid_yaml_files(self):
        """Loads two valid YAML files and returns dictionaries."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = _write_yaml(SAMPLE_KDES_A, tmpdir, "doc_a.yaml")
            path_b = _write_yaml(SAMPLE_KDES_B, tmpdir, "doc_b.yaml")

            kdes_1, kdes_2 = load_yaml_files(path_a, path_b)

            self.assertIsInstance(kdes_1, dict)
            self.assertIsInstance(kdes_2, dict)
            self.assertIn("element1", kdes_1)
            self.assertIn("element1", kdes_2)
            self.assertEqual(kdes_1["element1"]["name"], "Logging")
            self.assertEqual(kdes_2["element2"]["name"], "Pod Security")

    def test_raises_on_missing_file(self):
        """Raises FileNotFoundError for a nonexistent path."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path_a = _write_yaml(SAMPLE_KDES_A, tmpdir, "doc_a.yaml")
            with self.assertRaises(FileNotFoundError):
                load_yaml_files(path_a, "/nonexistent/file.yaml")

    def test_raises_on_non_yaml_file(self):
        """Raises ValueError for a non-YAML file."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not yaml")
            tmp_path = f.name
        try:
            with self.assertRaises(ValueError):
                load_yaml_files(tmp_path, tmp_path)
        finally:
            os.unlink(tmp_path)


class TestCompareElementNames(unittest.TestCase):
    """Test 2: compare_element_names"""

    def test_detects_name_differences(self):
        """Detects elements with different names across the two files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "diff_names.txt")
            result_path = compare_element_names(
                SAMPLE_KDES_A, SAMPLE_KDES_B, output_path=out_path
            )

            self.assertTrue(os.path.exists(result_path))
            content = open(result_path).read()

            # "Kubelet" is only in A, "Pod Security" is only in B
            self.assertIn("Kubelet", content)
            self.assertIn("Pod Security", content)
            # "Logging" is in both so should NOT appear
            self.assertNotIn("Logging", content)

    def test_reports_no_differences_when_identical(self):
        """Reports no differences when both files have the same names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "diff_names.txt")
            compare_element_names(
                SAMPLE_KDES_IDENTICAL, SAMPLE_KDES_IDENTICAL, output_path=out_path
            )

            content = open(out_path).read()
            self.assertIn("NO DIFFERENCES IN REGARDS TO ELEMENT NAMES", content)


class TestCompareElementRequirements(unittest.TestCase):
    """Test 3: compare_element_requirements"""

    def test_detects_requirement_differences(self):
        """Detects requirements that differ across the two files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "diff_reqs.txt")
            result_path = compare_element_requirements(
                SAMPLE_KDES_A, SAMPLE_KDES_B, output_path=out_path
            )

            self.assertTrue(os.path.exists(result_path))
            content = open(result_path).read()

            # Requirements unique to A
            self.assertIn("Kubelet,Ensure Anonymous Auth is Not Enabled", content)
            self.assertIn("Logging,Ensure audit logs are stored", content)
            # Requirements unique to B
            self.assertIn("Pod Security,Enforce restricted pod security standards", content)
            # Shared requirement should NOT appear
            self.assertNotIn("Logging,Enable audit Logs\n", content)

    def test_reports_no_differences_when_identical(self):
        """Reports no differences when both files have the same requirements."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "diff_reqs.txt")
            compare_element_requirements(
                SAMPLE_KDES_IDENTICAL, SAMPLE_KDES_IDENTICAL, output_path=out_path
            )

            content = open(out_path).read()
            self.assertIn("NO DIFFERENCES IN REGARDS TO ELEMENT REQUIREMENTS", content)


if __name__ == "__main__":
    unittest.main()
