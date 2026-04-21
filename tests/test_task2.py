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
            with open(result_path) as f:
                content = f.read()

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

            with open(out_path) as f:
                content = f.read()
            self.assertIn("NO DIFFERENCES IN REGARDS TO ELEMENT NAMES", content)

    def test_ignores_unparsed_llm_output_elements(self):
        """Elements named 'Unparsed LLM Output' must not appear in name diffs."""
        kdes_unparsed = {
            "element1": {"name": "Unparsed LLM Output", "requirements": ["raw escaped yaml..."]}
        }
        kdes_clean = {
            "element1": {"name": "Logging", "requirements": ["Enable audit logs"]}
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "diff_names.txt")
            result_path = compare_element_names(kdes_unparsed, kdes_clean, output_path=out_path)
            with open(result_path) as f:
                content = f.read()
            self.assertNotIn("Unparsed LLM Output", content)

    def test_ignores_garbage_element_names(self):
        """Non-ASCII names, placeholder 'KDEs', and meta-commentary must not appear in name diffs."""
        kdes_garbage = {
            "element1": {"name": "KDEs", "requirements": ["Enable audit logs"]},
            "element2": {"name": "\u0BBE\u0BB0\u0BC1", "requirements": ["Enable audit logs"]},
            "element3": {"name": "A helpful security document analyzer would identify the following KDEs",
                         "requirements": ["Enable audit logs"]},
        }
        kdes_clean = {"element1": {"name": "Logging", "requirements": ["Enable audit logs"]}}
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "diff_names.txt")
            result_path = compare_element_names(kdes_garbage, kdes_clean, output_path=out_path)
            with open(result_path, encoding="utf-8") as f:
                content = f.read()
            self.assertNotIn("KDEs", content)
            self.assertNotIn("\u0BBE\u0BB0\u0BC1", content)
            self.assertNotIn("helpful", content)


class TestCompareElementRequirements(unittest.TestCase):
    """Test 3: compare_element_requirements"""

    def test_detects_requirement_differences(self):
        """Detects name-only and requirement differences in spec-mandated format."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "diff_reqs.txt")
            result_path = compare_element_requirements(
                SAMPLE_KDES_A, SAMPLE_KDES_B,
                file_1_name="doc_a", file_2_name="doc_b",
                output_path=out_path,
            )

            self.assertTrue(os.path.exists(result_path))
            with open(result_path) as f:
                content = f.read()

            # Name-only diffs: Kubelet only in A, Pod Security only in B
            self.assertIn("Kubelet,ABSENT-IN-doc_b,PRESENT-IN-doc_a,NA", content)
            self.assertIn("Pod Security,ABSENT-IN-doc_a,PRESENT-IN-doc_b,NA", content)
            # Requirement diff: "Ensure audit logs are stored" only in A under Logging
            self.assertIn(
                "Logging,ABSENT-IN-doc_b,PRESENT-IN-doc_a,Ensure audit logs are stored",
                content,
            )
            # Shared requirement should NOT appear
            self.assertNotIn("Logging,ABSENT-IN-doc_b,PRESENT-IN-doc_a,Enable audit Logs\n", content)

    def test_reports_no_differences_when_identical(self):
        """Reports no differences when both files have the same requirements."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "diff_reqs.txt")
            compare_element_requirements(
                SAMPLE_KDES_IDENTICAL, SAMPLE_KDES_IDENTICAL,
                file_1_name="doc_a", file_2_name="doc_b",
                output_path=out_path,
            )

            with open(out_path) as f:
                content = f.read()
            self.assertIn("NO DIFFERENCES IN REGARDS TO ELEMENT REQUIREMENTS", content)

    def test_fuzzy_dedup_collapses_near_duplicate_requirements(self):
        """Requirements with different leading verbs but same substance should not appear in diffs."""
        kdes_a = {
            "element1": {
                "name": "RBAC",
                "requirements": [
                    "Avoid use of the Bind, Impersonate and Escalate permissions",
                    "Ensure that the --eventRecordQPS argument is set to 0 or a level",
                ],
            },
        }
        kdes_b = {
            "element1": {
                "name": "RBAC",
                "requirements": [
                    "Limit use of the Bind, Impersonate and Escalate permissions",
                    "Ensure that the --eventRecordQPS argument is set to 0 or a level which ensures appropriate event capture",
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "diff_reqs.txt")
            compare_element_requirements(
                kdes_a, kdes_b,
                file_1_name="doc_a", file_2_name="doc_b",
                output_path=out_path,
            )
            with open(out_path) as f:
                content = f.read()
            # "Bind, Impersonate and Escalate" requirements differ only in leading verb —
            # fuzzy dedup should collapse them
            self.assertNotIn("Bind, Impersonate", content)

    def test_ignores_unparsed_llm_output_requirements(self):
        """Elements named 'Unparsed LLM Output' must not appear in requirement diffs."""
        kdes_unparsed = {
            "element1": {"name": "Unparsed LLM Output", "requirements": ["raw escaped yaml..."]}
        }
        kdes_clean = {
            "element1": {"name": "Logging", "requirements": ["Enable audit logs"]}
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "diff_reqs.txt")
            result_path = compare_element_requirements(
                kdes_unparsed, kdes_clean,
                file_1_name="doc_a", file_2_name="doc_b",
                output_path=out_path,
            )
            with open(result_path) as f:
                content = f.read()
            self.assertNotIn("Unparsed LLM Output", content)


if __name__ == "__main__":
    unittest.main()
