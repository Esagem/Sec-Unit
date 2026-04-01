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
        self.assertIn("Step 1", prompt)
        self.assertIn("Step 2", prompt)


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


if __name__ == "__main__":
    unittest.main()
