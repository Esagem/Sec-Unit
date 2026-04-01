"""
Task 1: Extractor Module
=========================
Six functions for loading CIS Benchmark PDFs, constructing prompts,
extracting Key Data Elements (KDEs) via Gemma-3-1B, and logging output.
"""

import os
import yaml
import datetime
from pypdf import PdfReader

from task1.prompts import (
    build_zero_shot_prompt,
    build_few_shot_prompt,
    build_chain_of_thought_prompt,
)


# ---------------------------------------------------------------------------
# Function 1: load_documents
# ---------------------------------------------------------------------------
def load_documents(pdf_path_1: str, pdf_path_2: str) -> tuple[str, str]:
    """
    Loads two PDF documents, validates they can be opened, and extracts
    all text content from each.

    Args:
        pdf_path_1: File path to the first PDF.
        pdf_path_2: File path to the second PDF.

    Returns:
        A tuple of (text_1, text_2) containing the extracted text from
        each document.

    Raises:
        FileNotFoundError: If either path does not exist.
        ValueError: If either file is not a PDF or contains no extractable text.
    """
    texts = []
    for path in [pdf_path_1, pdf_path_2]:
        # --- Validation: existence ---
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Document not found: {path}")

        # --- Validation: extension ---
        if not path.lower().endswith(".pdf"):
            raise ValueError(f"Expected a PDF file, got: {path}")

        # --- Extraction ---
        try:
            reader = PdfReader(path)
        except Exception as e:
            raise ValueError(f"Cannot open PDF '{path}': {e}")

        if len(reader.pages) == 0:
            raise ValueError(f"PDF has no pages: {path}")

        full_text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"

        if not full_text.strip():
            raise ValueError(f"No extractable text in PDF: {path}")

        texts.append(full_text)

    return texts[0], texts[1]


# ---------------------------------------------------------------------------
# Function 2: construct_zero_shot_prompt
# ---------------------------------------------------------------------------
def construct_zero_shot_prompt(document_text: str) -> str:
    """
    Constructs a zero-shot prompt to identify Key Data Elements (KDEs)
    in a CIS Benchmark document.

    Args:
        document_text: The extracted text from a PDF document.

    Returns:
        A fully-formed prompt string ready to send to the LLM.
    """
    return build_zero_shot_prompt(document_text)


# ---------------------------------------------------------------------------
# Function 3: construct_few_shot_prompt
# ---------------------------------------------------------------------------
def construct_few_shot_prompt(document_text: str) -> str:
    """
    Constructs a few-shot prompt (with examples) to identify Key Data
    Elements (KDEs) in a CIS Benchmark document.

    Args:
        document_text: The extracted text from a PDF document.

    Returns:
        A fully-formed prompt string ready to send to the LLM.
    """
    return build_few_shot_prompt(document_text)


# ---------------------------------------------------------------------------
# Function 4: construct_chain_of_thought_prompt
# ---------------------------------------------------------------------------
def construct_chain_of_thought_prompt(document_text: str) -> str:
    """
    Constructs a chain-of-thought prompt to identify Key Data Elements
    (KDEs) in a CIS Benchmark document.

    Args:
        document_text: The extracted text from a PDF document.

    Returns:
        A fully-formed prompt string ready to send to the LLM.
    """
    return build_chain_of_thought_prompt(document_text)


# ---------------------------------------------------------------------------
# Function 5: extract_kdes
# ---------------------------------------------------------------------------
def extract_kdes(
    document_text: str,
    prompt_type: str,
    pipe,
    doc_name: str,
    output_dir: str = "outputs",
    max_new_tokens: int = 2048,
) -> dict:
    """
    Uses the specified prompt type to identify KDEs in the document text
    via the Gemma-3-1B pipeline, parses the response into a nested dict,
    and saves the result as a YAML file.

    Args:
        document_text: Extracted text from a single PDF.
        prompt_type: One of 'zero_shot', 'few_shot', 'chain_of_thought'.
        pipe: A HuggingFace text-generation pipeline (Gemma-3-1B).
        doc_name: Base name of the source PDF (e.g. 'cis-r1').
        output_dir: Directory to write YAML output.
        max_new_tokens: Max tokens for generation.

    Returns:
        A dict with structure:
            { "element1": { "name": ..., "requirements": [...] }, ... }
    """
    # --- Truncate text to fit model context ---------------------------------
    # Gemma-3-1B has a limited context window; we keep the most relevant
    # section (Recommendations typically start after the table of contents).
    truncated = _truncate_to_recommendations(document_text)

    # --- Build prompt -------------------------------------------------------
    prompt_builders = {
        "zero_shot": construct_zero_shot_prompt,
        "few_shot": construct_few_shot_prompt,
        "chain_of_thought": construct_chain_of_thought_prompt,
    }
    if prompt_type not in prompt_builders:
        raise ValueError(
            f"Invalid prompt_type '{prompt_type}'. "
            f"Must be one of: {list(prompt_builders.keys())}"
        )

    prompt_text = prompt_builders[prompt_type](truncated)

    # --- Call the LLM -------------------------------------------------------
    messages = [
        [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": "You are a helpful security document analyzer.",
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}],
            },
        ]
    ]

    output = pipe(messages)
    raw_response = output[0][0]["generated_text"][-1]["content"]

    # --- Parse response into nested dict ------------------------------------
    kdes = _parse_kdes_from_response(raw_response)

    # --- Save YAML ----------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    yaml_filename = f"{doc_name}-kdes.yaml"
    yaml_path = os.path.join(output_dir, yaml_filename)
    with open(yaml_path, "w") as f:
        yaml.dump(kdes, f, default_flow_style=False, sort_keys=False)

    return {
        "kdes": kdes,
        "raw_response": raw_response,
        "prompt_used": prompt_text,
        "prompt_type": prompt_type,
        "yaml_path": yaml_path,
    }


# ---------------------------------------------------------------------------
# Function 6: collect_llm_output
# ---------------------------------------------------------------------------
def collect_llm_output(
    results: list[dict],
    output_path: str = "outputs/llm_output.txt",
    llm_name: str = "google/gemma-3-1b-it",
) -> str:
    """
    Collects the output from all LLM runs and writes a formatted TEXT file.

    Args:
        results: A list of dicts returned by extract_kdes(), each containing
                 'raw_response', 'prompt_used', and 'prompt_type'.
        output_path: Path to the output text file.
        llm_name: Name of the LLM used.

    Returns:
        The path to the written output file.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        f.write(f"LLM Output Collection — {datetime.datetime.now().isoformat()}\n")
        f.write("=" * 70 + "\n\n")

        for i, result in enumerate(results, 1):
            f.write(f"--- Run {i} ---\n\n")
            f.write(f"*LLM Name*\n{llm_name}\n\n")
            f.write(f"*Prompt Used*\n{result['prompt_used']}\n\n")
            f.write(f"*Prompt Type*\n{result['prompt_type']}\n\n")
            f.write(f"*LLM Output*\n{result['raw_response']}\n\n")
            f.write("=" * 70 + "\n\n")

    return output_path


# ---------------------------------------------------------------------------
# Batch-friendly helpers
# ---------------------------------------------------------------------------
def build_messages(document_text: str, prompt_type: str) -> tuple[list, str]:
    """
    Build the messages list and prompt text for a single doc+prompt_type pair
    without calling the pipeline. Used for batched inference.

    Returns:
        (messages, prompt_text) where messages is ready to pass to pipe().
    """
    prompt_builders = {
        "zero_shot": construct_zero_shot_prompt,
        "few_shot": construct_few_shot_prompt,
        "chain_of_thought": construct_chain_of_thought_prompt,
    }
    if prompt_type not in prompt_builders:
        raise ValueError(
            f"Invalid prompt_type '{prompt_type}'. "
            f"Must be one of: {list(prompt_builders.keys())}"
        )
    truncated = _truncate_to_recommendations(document_text)
    prompt_text = prompt_builders[prompt_type](truncated)
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a helpful security document analyzer."}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": prompt_text}],
        },
    ]
    return messages, prompt_text


def save_kde_result(
    raw_response: str,
    prompt_text: str,
    prompt_type: str,
    doc_name: str,
    output_dir: str = "outputs",
) -> dict:
    """
    Parse a raw LLM response into KDEs, write the YAML file, and return
    the result dict. Used after batched inference to post-process outputs.
    """
    kdes = _parse_kdes_from_response(raw_response)
    os.makedirs(output_dir, exist_ok=True)
    yaml_path = os.path.join(output_dir, f"{doc_name}-kdes.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(kdes, f, default_flow_style=False, sort_keys=False)
    return {
        "kdes": kdes,
        "raw_response": raw_response,
        "prompt_used": prompt_text,
        "prompt_type": prompt_type,
        "yaml_path": yaml_path,
    }


# ===========================================================================
# Internal helpers
# ===========================================================================
def _truncate_to_recommendations(text: str, max_chars: int = 6000) -> str:
    """
    Attempts to isolate the 'Recommendations' section from a CIS Benchmark
    document and truncates to fit within the model's practical context.
    """
    # Try to find the recommendations section
    markers = ["Recommendations\n", "1 Control Plane Components"]
    start = 0
    for marker in markers:
        idx = text.find(marker)
        if idx != -1:
            start = idx
            break

    section = text[start:]
    if len(section) > max_chars:
        section = section[:max_chars] + "\n\n[... truncated ...]"
    return section


def _parse_kdes_from_response(response: str) -> dict:
    """
    Parses the LLM's raw text response into a nested dict of KDEs.

    Tries YAML parsing first. Falls back to a line-by-line heuristic parser
    that looks for element/requirement patterns in the text.
    """
    # --- Attempt 1: Direct YAML parse (if model returned valid YAML) --------
    # Strip markdown code fences if present
    cleaned = response.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        # Remove first and last fence lines
        lines = [l for l in lines if not l.strip().startswith("```")]
        cleaned = "\n".join(lines)

    try:
        parsed = yaml.safe_load(cleaned)
        if isinstance(parsed, dict):
            return _normalize_kdes(parsed)
    except yaml.YAMLError:
        pass

    # --- Attempt 2: Heuristic line-by-line parsing --------------------------
    kdes = {}
    current_element = None
    element_counter = 0

    for line in response.split("\n"):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        # Detect element headers (numbered sections like "3.2.1 Ensure...")
        if _looks_like_element_header(stripped):
            element_counter += 1
            element_key = f"element{element_counter}"
            current_element = {
                "name": stripped,
                "requirements": [],
            }
            kdes[element_key] = current_element
        elif current_element is not None:
            # Treat subsequent non-empty lines as requirements
            if stripped.startswith("- "):
                stripped = stripped[2:]
            if stripped:
                current_element["requirements"].append(stripped)

    # If heuristic also found nothing, store the raw response as a single element
    if not kdes:
        kdes["element1"] = {
            "name": "Unparsed LLM Output",
            "requirements": [response.strip()],
        }

    return kdes


def _looks_like_element_header(line: str) -> bool:
    """Check if a line looks like a CIS benchmark section header."""
    import re

    # Matches patterns like "3.2.1 Ensure..." or "1.1 Enable..."
    return bool(re.match(r"^\d+(\.\d+)+ ", line))


def _normalize_kdes(data: dict) -> dict:
    """
    Normalizes a parsed YAML dict into the expected KDE schema:
    { elementN: { name: str, requirements: [str, ...] } }
    """
    normalized = {}
    counter = 0

    for key, value in data.items():
        counter += 1
        elem_key = f"element{counter}"

        if isinstance(value, dict):
            name = value.get("name", str(key))
            reqs = value.get("requirements", [])
            if isinstance(reqs, str):
                reqs = [reqs]
            elif not isinstance(reqs, list):
                reqs = [str(reqs)]
            normalized[elem_key] = {"name": str(name), "requirements": reqs}
        elif isinstance(value, list):
            normalized[elem_key] = {"name": str(key), "requirements": value}
        else:
            normalized[elem_key] = {
                "name": str(key),
                "requirements": [str(value)],
            }

    return normalized
