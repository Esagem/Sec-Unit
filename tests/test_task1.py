"""
Test cases for Task 1: Extractor
=================================
One test case per function (6 total), as required by the project spec.
Tests are designed to run without a GPU or the Gemma model loaded.
"""

import os
import sys
import yaml
import tempfile
import unittest
from unittest.mock import MagicMock, patch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from task1.extractor import (
    load_documents,
    construct_zero_shot_prompt,
    construct_few_shot_prompt,
    construct_chain_of_thought_prompt,
    extract_kdes,
    collect_llm_output,
)


# Path to real test PDFs (adjust if running from different directory)
INPUTS_DIR = os.path.join(os.path.dirname(__file__), "..", "inputs")
PDF_1 = os.path.join(INPUTS_DIR, "cis-r1.pdf")
PDF_2 = os.path.join(INPUTS_DIR, "cis-r2.pdf")


class TestLoadDocuments(unittest.TestCase):
    """Test 1: load_documents"""

    def test_loads_two_valid_pdfs(self):
        """Loads two real CIS PDFs and returns non-empty text for each."""
        if not os.path.exists(PDF_1) or not os.path.exists(PDF_2):
            self.skipTest("Test PDFs not found in inputs/ directory")

        text_1, text_2 = load_documents(PDF_1, PDF_2)

        # Both should return substantial text
        self.assertIsInstance(text_1, str)
        self.assertIsInstance(text_2, str)
        self.assertGreater(len(text_1), 1000)
        self.assertGreater(len(text_2), 1000)

    def test_raises_on_missing_file(self):
        """Raises FileNotFoundError for a nonexistent path."""
        with self.assertRaises(FileNotFoundError):
            load_documents("/nonexistent/file.pdf", PDF_1 if os.path.exists(PDF_1) else "/also/fake.pdf")

    def test_raises_on_non_pdf(self):
        """Raises ValueError if a file is not a PDF."""
        # Create a temp .txt file
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"not a pdf")
            tmp_path = f.name
        try:
            with self.assertRaises(ValueError):
                load_documents(tmp_path, tmp_path)
        finally:
            os.unlink(tmp_path)


class TestConstructZeroShotPrompt(unittest.TestCase):
    """Test 2: construct_zero_shot_prompt"""

    def test_returns_prompt_with_document_text(self):
        """Prompt should contain the document text and KDE instructions."""
        sample_text = "3.1.1 Ensure that anonymous auth is disabled"
        prompt = construct_zero_shot_prompt(sample_text)

        self.assertIsInstance(prompt, str)
        self.assertIn(sample_text, prompt)
        self.assertIn("Key Data Element", prompt)
        self.assertIn("YAML", prompt)


class TestConstructFewShotPrompt(unittest.TestCase):
    """Test 3: construct_few_shot_prompt"""

    def test_returns_prompt_with_examples_and_document(self):
        """Few-shot prompt should include examples and the document text."""
        sample_text = "4.1.1 Ensure default namespace is not used"
        prompt = construct_few_shot_prompt(sample_text)

        self.assertIsInstance(prompt, str)
        self.assertIn(sample_text, prompt)
        # Should contain example input/output pairs
        self.assertIn("EXAMPLE INPUT", prompt)
        self.assertIn("EXAMPLE OUTPUT", prompt)
        self.assertIn("element1", prompt)


class TestConstructChainOfThoughtPrompt(unittest.TestCase):
    """Test 4: construct_chain_of_thought_prompt"""

    def test_returns_prompt_with_reasoning_steps(self):
        """CoT prompt should instruct step-by-step reasoning."""
        sample_text = "2.1.1 Enable audit logs"
        prompt = construct_chain_of_thought_prompt(sample_text)

        self.assertIsInstance(prompt, str)
        self.assertIn(sample_text, prompt)
        self.assertIn("think through", prompt)
        self.assertIn("Output ONLY", prompt)


class TestExtractKDEs(unittest.TestCase):
    """Test 5: extract_kdes"""

    def test_extract_kdes_with_mock_pipeline(self):
        """
        Mocks the Gemma pipeline and verifies KDE extraction produces
        a valid YAML file with the expected structure.
        """
        # Simulate LLM returning valid YAML
        mock_yaml_response = (
            'element1:\n'
            '  name: "Logging"\n'
            '  requirements:\n'
            '    - "Enable audit Logs"\n'
            'element2:\n'
            '  name: "Kubelet"\n'
            '  requirements:\n'
            '    - "Ensure Anonymous Auth is Not Enabled"\n'
        )

        # Build a mock pipeline that returns this response
        mock_pipe = MagicMock()
        mock_pipe.return_value = [
            [
                {
                    "generated_text": [
                        {"role": "system", "content": "..."},
                        {"role": "user", "content": "..."},
                        {"role": "assistant", "content": mock_yaml_response},
                    ]
                }
            ]
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            result = extract_kdes(
                document_text="2.1.1 Enable audit Logs\n3.2.1 Ensure Anonymous Auth is Not Enabled",
                prompt_type="zero_shot",
                pipe=mock_pipe,
                doc_name="cis-r1",
                output_dir=tmpdir,
            )

            # Check return structure
            self.assertIn("kdes", result)
            self.assertIn("raw_response", result)
            self.assertIn("yaml_path", result)
            self.assertIn("prompt_type", result)

            # Check YAML file was created
            self.assertTrue(os.path.exists(result["yaml_path"]))

            # Check YAML contents
            with open(result["yaml_path"]) as f:
                loaded = yaml.safe_load(f)
            self.assertIn("element1", loaded)
            self.assertEqual(loaded["element1"]["name"], "Logging")


class TestCollectLLMOutput(unittest.TestCase):
    """Test 6: collect_llm_output"""

    def test_writes_formatted_output_file(self):
        """Collects multiple results into a single formatted TEXT file."""
        mock_results = [
            {
                "prompt_used": "Analyze this document...",
                "prompt_type": "zero_shot",
                "raw_response": "element1:\n  name: Logging\n  requirements:\n    - Enable audit Logs",
            },
            {
                "prompt_used": "Given these examples...",
                "prompt_type": "few_shot",
                "raw_response": "element1:\n  name: Kubelet\n  requirements:\n    - Disable anonymous auth",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "llm_output.txt")
            result_path = collect_llm_output(mock_results, output_path=out_path)

            self.assertTrue(os.path.exists(result_path))

            with open(result_path) as f:
                content = f.read()
            self.assertIn("*LLM Name*", content)
            self.assertIn("google/gemma-3-1b-it", content)
            self.assertIn("*Prompt Type*", content)
            self.assertIn("zero_shot", content)
            self.assertIn("few_shot", content)
            self.assertIn("*LLM Output*", content)


class TestFenceStripping(unittest.TestCase):
    """Test that _parse_kdes_from_response correctly strips fences with preamble text."""

    def test_extracts_yaml_between_fences_with_preamble(self):
        """Response with text before opening fence should still parse correctly."""
        from task1.extractor import _parse_kdes_from_response

        response = (
            "Here is the extracted YAML output:\n"
            "```yaml\n"
            "element1:\n"
            "  name: \"Logging\"\n"
            "  requirements:\n"
            "    - \"Enable audit Logs\"\n"
            "```\n"
        )
        result = _parse_kdes_from_response(response)
        self.assertIn("element1", result)
        self.assertEqual(result["element1"]["name"], "Logging")
        self.assertNotEqual(result["element1"]["name"], "Unparsed LLM Output")

    def test_extracts_yaml_between_fences_no_preamble(self):
        """Standard response starting directly with fence should parse correctly."""
        from task1.extractor import _parse_kdes_from_response

        response = (
            "```yaml\n"
            "element1:\n"
            "  name: \"Pod Security\"\n"
            "  requirements:\n"
            "    - \"Enforce restricted standards\"\n"
            "```\n"
        )
        result = _parse_kdes_from_response(response)
        self.assertIn("element1", result)
        self.assertEqual(result["element1"]["name"], "Pod Security")


class TestPartialRecovery(unittest.TestCase):
    """Test that _parse_kdes_from_response recovers valid elements from truncated output."""

    def test_recovers_complete_elements_before_truncation(self):
        """Complete elements before a truncated final element should be recovered."""
        from task1.extractor import _parse_kdes_from_response

        # Simulates model hitting token limit mid-string in element2
        truncated_response = (
            'element1:\n'
            '  name: "Logging"\n'
            '  requirements:\n'
            '    - "Enable audit logs"\n'
            'element2:\n'
            '  name: "Kubelet"\n'
            '  requirements:\n'
            '    - "Ensure anonymous auth is not enabled\n'
        )
        result = _parse_kdes_from_response(truncated_response)
        self.assertIn("element1", result)
        self.assertEqual(result["element1"]["name"], "Logging")
        self.assertNotIn("Unparsed LLM Output",
                         [v.get("name") for v in result.values() if isinstance(v, dict)])


class TestElementNameValidation(unittest.TestCase):
    """_normalize_kdes must reject garbage element names and single-word requirements."""

    def test_rejects_non_ascii_element_name(self):
        """Elements with non-ASCII names must be dropped."""
        from task1.extractor import _parse_kdes_from_response
        response = 'element1:\n  name: "\u0BBE\u0BB0\u0BC1"\n  requirements:\n    - "Ensure X is set"\n'
        result = _parse_kdes_from_response(response)
        names = [v.get("name") for v in result.values() if isinstance(v, dict)]
        self.assertNotIn("\u0BBE\u0BB0\u0BC1", names)

    def test_rejects_placeholder_name_kdes(self):
        """Elements named 'KDEs' must be dropped."""
        from task1.extractor import _parse_kdes_from_response
        response = 'element1:\n  name: "KDEs"\n  requirements:\n    - "Ensure X is set"\n'
        result = _parse_kdes_from_response(response)
        names = [v.get("name") for v in result.values() if isinstance(v, dict)]
        self.assertNotIn("KDEs", names)

    def test_rejects_meta_commentary_element_name(self):
        """Elements with meta-commentary names (> 60 chars or containing 'helpful') must be dropped."""
        from task1.extractor import _parse_kdes_from_response
        long_name = "A helpful security document analyzer would identify the following KDEs"
        response = f'element1:\n  name: "{long_name}"\n  requirements:\n    - "Ensure X is set"\n'
        result = _parse_kdes_from_response(response)
        names = [v.get("name") for v in result.values() if isinstance(v, dict)]
        self.assertNotIn(long_name, names)

    def test_rejects_single_word_requirements(self):
        """Single-word entries like '3' and 'Automated' must be filtered from requirements."""
        from task1.extractor import _normalize_kdes
        data = {
            "element1": {
                "name": "Logging",
                "requirements": ["Enable audit logs", "3", "Automated", "Manual"],
            }
        }
        result = _normalize_kdes(data)
        reqs = result["element1"]["requirements"]
        self.assertIn("Enable audit logs", reqs)
        self.assertNotIn("3", reqs)
        self.assertNotIn("Automated", reqs)
        self.assertNotIn("Manual", reqs)

    def test_valid_element_names_pass_through(self):
        """Legitimate CIS section names must not be filtered."""
        from task1.extractor import _normalize_kdes
        data = {
            "element1": {"name": "Control Plane Components", "requirements": ["Ensure X is set"]},
            "element2": {"name": "Kubelet", "requirements": ["Ensure Y is enabled"]},
        }
        result = _normalize_kdes(data)
        names = [v["name"] for v in result.values()]
        self.assertIn("Control Plane Components", names)
        self.assertIn("Kubelet", names)


class TestChunkDocument(unittest.TestCase):
    """_chunk_document must cover the entire document — not truncate it."""

    def test_short_document_returns_single_chunk(self):
        from task1.extractor import _chunk_document
        text = "3.2.1 Ensure anonymous auth is disabled\n"
        chunks = _chunk_document(text, max_chars=20000)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0], text)

    def test_long_document_is_fully_covered(self):
        """Every character of the input must appear in at least one chunk."""
        from task1.extractor import _chunk_document
        # Build a realistic multi-section body ~60k chars
        lines = []
        for section in range(1, 7):
            lines.append(f"{section} Section {section}")
            for sub in range(1, 30):
                lines.append(f"{section}.{sub}.1 Ensure that something important is configured correctly for item {sub}")
                lines.append("Rationale: " + ("x " * 80))
                lines.append("Remediation: " + ("y " * 80))
        text = "\n".join(lines)
        self.assertGreater(len(text), 50000)

        chunks = _chunk_document(text, max_chars=15000, overlap_chars=500)
        self.assertGreater(len(chunks), 1, "long doc must split into multiple chunks")

        # Every chunk respects the cap
        for c in chunks:
            self.assertLessEqual(len(c), 15000)

        # Joining chunks (accounting for overlap) covers the full document —
        # the simplest correctness check: content from the last section must
        # appear in some chunk.
        last_section_marker = "Section 6"
        self.assertTrue(
            any(last_section_marker in c for c in chunks),
            "content from the end of the document must appear in a chunk",
        )

        # Content from the middle must also appear
        mid_marker = "Section 3"
        self.assertTrue(any(mid_marker in c for c in chunks))

    def test_chunks_overlap_for_continuity(self):
        """Adjacent chunks should share some tail/head content so controls
        spanning a boundary appear intact in at least one chunk."""
        from task1.extractor import _chunk_document
        text = "A" * 30000
        chunks = _chunk_document(text, max_chars=10000, overlap_chars=500)
        self.assertGreaterEqual(len(chunks), 3)
        # End of chunk[0] should share a suffix with start of chunk[1]
        overlap = chunks[0][-500:]
        self.assertIn(overlap[:100], chunks[1])


class TestMergeKdeDicts(unittest.TestCase):
    """_merge_kde_dicts must union requirements across chunks by element name."""

    def test_same_name_across_chunks_is_merged(self):
        from task1.extractor import _merge_kde_dicts
        chunk_a = {
            "element1": {"name": "Kubelet", "requirements": ["Ensure anonymous auth is disabled"]},
        }
        chunk_b = {
            "element1": {"name": "Kubelet", "requirements": ["Ensure read-only port is disabled"]},
            "element2": {"name": "Logging", "requirements": ["Enable audit logs"]},
        }
        merged = _merge_kde_dicts([chunk_a, chunk_b])
        names = [v["name"] for v in merged.values()]
        self.assertEqual(sorted(names), ["Kubelet", "Logging"])
        kubelet = next(v for v in merged.values() if v["name"] == "Kubelet")
        self.assertIn("Ensure anonymous auth is disabled", kubelet["requirements"])
        self.assertIn("Ensure read-only port is disabled", kubelet["requirements"])


if __name__ == "__main__":
    unittest.main()
