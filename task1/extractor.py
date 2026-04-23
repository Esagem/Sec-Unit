"""
Task 1: Extractor Module
=========================
Six functions for loading CIS Benchmark PDFs, constructing prompts,
extracting Key Data Elements (KDEs) via Gemma-3-1B, and logging output.
"""

import os
import re
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
    via the Gemma-3-1B pipeline. Chunks the document to cover content past
    the model's context window, runs inference on each chunk, and merges
    the resulting KDEs (same-name elements have their requirements unioned).
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

    chunks = _chunk_document(document_text)

    raw_responses: list[str] = []
    parsed_chunks: list[dict] = []
    representative_prompt: str | None = None

    for chunk in chunks:
        prompt_text = prompt_builders[prompt_type](chunk)
        if representative_prompt is None:
            representative_prompt = prompt_text
        messages = [
            [
                {
                    "role": "system",
                    "content": [
                        {
                            "type": "text",
                            "text": "You are a YAML formatter. Output only valid YAML. Do not write any text before or after the YAML block.",
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
        raw = output[0][0]["generated_text"][-1]["content"]
        raw_responses.append(raw)
        parsed_chunks.append(_parse_kdes_from_response(raw))

    kdes = _merge_kde_dicts(parsed_chunks)

    os.makedirs(output_dir, exist_ok=True)
    yaml_path = os.path.join(output_dir, f"{doc_name}-kdes.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(kdes, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return {
        "kdes": kdes,
        "raw_response": _join_chunk_responses(raw_responses),
        "prompt_used": representative_prompt or "",
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

    with open(output_path, "w", encoding="utf-8") as f:
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
def build_messages(document_text: str, prompt_type: str) -> list[tuple[list, str]]:
    """
    Build one (messages, prompt_text) pair per chunk of the document. Callers
    flatten the list across all (doc, prompt) jobs into a single batched
    pipeline call, then regroup the responses by chunk-group to merge.

    Returns:
        A list of (messages, prompt_text) tuples — one per chunk.
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
    chunks = _chunk_document(document_text)
    out: list[tuple[list, str]] = []
    for chunk in chunks:
        prompt_text = prompt_builders[prompt_type](chunk)
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
        out.append((messages, prompt_text))
    return out


def save_kde_result(
    raw_response,
    prompt_text: str,
    prompt_type: str,
    doc_name: str,
    output_dir: str = "outputs",
) -> dict:
    """
    Parse raw LLM response(s) into KDEs, write the YAML file, and return
    the result dict. `raw_response` may be a single string (unchunked input)
    or a list of strings — one per chunk — whose parsed KDEs are merged by name.
    """
    if isinstance(raw_response, list):
        parsed_chunks = [_parse_kdes_from_response(r) for r in raw_response]
        kdes = _merge_kde_dicts(parsed_chunks)
        raw_joined = _join_chunk_responses(raw_response)
    else:
        kdes = _parse_kdes_from_response(raw_response)
        raw_joined = raw_response
    os.makedirs(output_dir, exist_ok=True)
    yaml_path = os.path.join(output_dir, f"{doc_name}-kdes.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.dump(kdes, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return {
        "kdes": kdes,
        "raw_response": raw_joined,
        "prompt_used": prompt_text,
        "prompt_type": prompt_type,
        "yaml_path": yaml_path,
    }


# ===========================================================================
# Internal helpers
# ===========================================================================
# Target size for a single chunk. Gemma-3-1B has a 32k-token context; at
# ~4 chars/token a 20k-char chunk is ~5k input tokens, leaving comfortable
# headroom for prompt wrapping (few-shot examples add ~2k chars) plus the
# 3072-token generation budget. Overlap preserves continuity across chunk
# boundaries so a control whose description spans a boundary still appears
# intact in at least one chunk.
_DEFAULT_CHUNK_CHARS = int(os.environ.get("SEC_UNIT_CHUNK_CHARS", "20000"))
_DEFAULT_OVERLAP_CHARS = int(os.environ.get("SEC_UNIT_CHUNK_OVERLAP", "800"))


def _chunk_document(
    text: str,
    max_chars: int = _DEFAULT_CHUNK_CHARS,
    overlap_chars: int = _DEFAULT_OVERLAP_CHARS,
) -> list[str]:
    """
    Split the full document into overlapping chunks that each fit within
    max_chars. Prefers to cut at a newline in the last 20% of each window
    so control descriptions aren't sliced mid-line.
    """
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    pos = 0
    while pos < len(text):
        end = min(pos + max_chars, len(text))
        if end < len(text):
            window_start = pos + int(max_chars * 0.8)
            nl = text.rfind("\n", window_start, end)
            if nl > pos:
                end = nl
        chunks.append(text[pos:end])
        if end >= len(text):
            break
        pos = max(end - overlap_chars, pos + 1)
    return chunks


def _merge_kde_dicts(dicts: list[dict]) -> dict:
    """
    Combine multiple parsed KDE dicts into one. Elements sharing the same
    name across chunks have their requirements unioned (dedup preserves order).
    Relies on `_normalize_kdes` for the by-name merge pass.
    """
    merged: dict[str, dict] = {}
    counter = 0
    for d in dicts:
        if not isinstance(d, dict):
            continue
        for v in d.values():
            if isinstance(v, dict) and "name" in v:
                counter += 1
                merged[f"element{counter}"] = v
    return _normalize_kdes(merged)


def _join_chunk_responses(responses: list[str]) -> str:
    """Concatenate per-chunk raw responses for inclusion in the llm_output log."""
    if len(responses) == 1:
        return responses[0]
    return "\n\n---CHUNK---\n\n".join(responses)


def _parse_kdes_from_response(response: str) -> dict:
    """
    Parses the LLM's raw text response into a nested dict of KDEs.

    Tries YAML parsing first. Falls back to a line-by-line heuristic parser
    that looks for element/requirement patterns in the text.
    """
    # --- Attempt 1: Direct YAML parse (if model returned valid YAML) --------
    # Strip markdown code fences if present
    cleaned = response.strip()
    if "```" in cleaned:
        lines = cleaned.split("\n")
        fence_indices = [i for i, l in enumerate(lines) if l.strip().startswith("```")]
        if len(fence_indices) >= 2:
            cleaned = "\n".join(lines[fence_indices[0] + 1 : fence_indices[-1]])
        elif len(fence_indices) == 1:
            cleaned = "\n".join(lines[fence_indices[0] + 1 :])

    try:
        parsed = yaml.safe_load(cleaned)
        if isinstance(parsed, dict):
            return _normalize_kdes(parsed)
    except yaml.YAMLError:
        pass

    # --- Attempt 1b: Partial recovery — truncate at last complete element ----
    lines_cleaned = cleaned.split("\n")
    element_starts = [
        i for i, l in enumerate(lines_cleaned) if re.match(r'^element\d+:', l.strip())
    ]
    if len(element_starts) >= 2:
        for cutoff in reversed(element_starts[1:]):
            partial = "\n".join(lines_cleaned[:cutoff])
            try:
                parsed = yaml.safe_load(partial)
                if isinstance(parsed, dict) and parsed:
                    return _normalize_kdes(parsed)
            except yaml.YAMLError:
                continue

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
            if stripped and stripped not in current_element["requirements"] and len(stripped.split()) >= 2:
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
    # Matches patterns like "3.2.1 Ensure..." or "1.1 Enable..."
    return bool(re.match(r"^\d+(\.\d+)+ ", line))


_INVALID_NAME_FRAGMENTS = {"helpful", "analyzer", "identify", "following", "document"}
_PLACEHOLDER_NAMES = {"kdes", "none"}
# CIS per-control sub-headings (appear on every control page, not KDEs themselves)
_BOILERPLATE_NAMES = {
    "title", "description", "profile applicability", "assessment status",
    "rationale", "rationale statement", "audit", "audit procedure",
    "remediation", "remediation procedure", "impact", "default value",
    "references", "cis controls",
}
# Imperative verbs start requirements, not section names. CIS top-level
# sections are all noun phrases (Kubelet, Logging, Worker Nodes, etc.).
_IMPERATIVE_VERB_RE = re.compile(
    r'^(ensure|verify|confirm|disable|enable|configure|minimize|avoid|limit|restrict|prefer|apply|use)\b',
    re.IGNORECASE,
)
_NOISE_PATTERNS = [
    re.compile(r'^page\s+\d+', re.IGNORECASE),           # page footer
    re.compile(r'^recommendation\s+\d', re.IGNORECASE),  # changelog entry
    re.compile(r'^unparsed\s+llm\s+output', re.IGNORECASE),
    re.compile(r'^element\s*\d+$', re.IGNORECASE),       # leaked element key
]


def _is_valid_element_name(name: str) -> bool:
    """Return True only for plausible CIS section names."""
    s = str(name).strip()
    if not s.isascii():
        return False
    if len(s) < 3 or len(s) > 60:
        return False
    low = s.lower()
    if low in _PLACEHOLDER_NAMES:
        return False
    if low in _BOILERPLATE_NAMES:
        return False
    if any(frag in low for frag in _INVALID_NAME_FRAGMENTS):
        return False
    if _IMPERATIVE_VERB_RE.match(s):
        return False
    if any(p.search(s) for p in _NOISE_PATTERNS):
        return False
    return True


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
            reqs = [re.sub(r'\s*\((Manual|Automated)\)\s*$', '', str(r)).strip() for r in reqs]
            # Strip leading CIS section numbers (e.g. "3.2.6 Ensure..." → "Ensure...")
            reqs = [re.sub(r'^\d+(\.\d+)+\s+', '', r).strip() for r in reqs]
            reqs = list(dict.fromkeys(reqs))
            # Reject short noise and truncated fragments
            reqs = [r for r in reqs if len(str(r).strip().split()) >= 3]
            # Reject requirements that end mid-phrase (trailing article/preposition)
            _FRAG_ENDINGS = {"the", "a", "an", "of", "for", "in", "with", "by", "to", "that", "from", "which", "and", "or", "not", "is"}
            reqs = [r for r in reqs if r.strip().split()[-1].lower() not in _FRAG_ENDINGS]
            if _is_valid_element_name(name):
                normalized[elem_key] = {"name": str(name), "requirements": reqs}
        elif isinstance(value, list):
            normalized[elem_key] = {"name": str(key), "requirements": value}
        else:
            normalized[elem_key] = {
                "name": str(key),
                "requirements": [str(value)],
            }

    # Merge elements sharing the same name (combines and deduplicates requirements)
    seen: dict[str, dict] = {}
    for elem_val in normalized.values():
        name = elem_val["name"]
        if name in seen:
            combined = list(dict.fromkeys(seen[name]["requirements"] + elem_val["requirements"]))
            seen[name]["requirements"] = combined
        else:
            seen[name] = elem_val
    # Drop elements with no requirements (empty extraction artifacts)
    return {f"element{i+1}": v for i, v in enumerate(seen.values()) if v["requirements"]}
