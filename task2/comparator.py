"""
Task 2: Comparator Module
==========================
Three functions for loading KDE YAML files from Task 1 and identifying
differences in element names and requirements between two documents.
"""

import os
import re
from difflib import SequenceMatcher

import yaml

_INVALID_KDE_NAMES = {"unparsed llm output", "kdes", "none"}
_INVALID_KDE_FRAGMENTS = {"helpful", "analyzer", "identify", "following"}

# Leading verbs that CIS benchmarks use interchangeably
_LEADING_VERB_RE = re.compile(
    r'^(Ensure that the|Ensure that|Ensure the|Ensure|Verify that the|Verify that|'
    r'Verify the|Verify|Confirm that|Confirm|Avoid use of the|Avoid use of|'
    r'Limit use of the|Limit use of|Minimize|Enable|Disable)\s+',
    re.IGNORECASE,
)


def _normalize_req(req: str) -> str:
    """Return a canonical form of a requirement for fuzzy comparison."""
    s = req.strip().rstrip(".")
    # Strip leading CIS verb phrase
    s = _LEADING_VERB_RE.sub("", s)
    # Collapse whitespace and lowercase
    s = " ".join(s.lower().split())
    return s


def _fuzzy_dedup_pairs(
    pairs_1: set[tuple[str, str]],
    pairs_2: set[tuple[str, str]],
    threshold: float = 0.85,
) -> set[tuple[str, str]]:
    """
    Symmetric difference of (name, requirement) pairs, but treats two
    requirements under the same element name as identical when their
    normalized forms match or their similarity ratio exceeds *threshold*.
    """
    # Exact symmetric difference first
    only_1 = pairs_1 - pairs_2
    only_2 = pairs_2 - pairs_1
    if not only_1 or not only_2:
        return only_1 | only_2

    # Group by element name for efficient comparison
    from collections import defaultdict
    by_name_1: dict[str, list[str]] = defaultdict(list)
    by_name_2: dict[str, list[str]] = defaultdict(list)
    for name, req in only_1:
        by_name_1[name].append(req)
    for name, req in only_2:
        by_name_2[name].append(req)

    matched_1: set[tuple[str, str]] = set()
    matched_2: set[tuple[str, str]] = set()

    for name in by_name_1:
        if name not in by_name_2:
            continue
        for r1 in by_name_1[name]:
            n1 = _normalize_req(r1)
            for r2 in by_name_2[name]:
                if (name, r2) in matched_2:
                    continue
                n2 = _normalize_req(r2)
                # Exact normalized match or high similarity
                if n1 == n2 or SequenceMatcher(None, n1, n2).ratio() >= threshold:
                    matched_1.add((name, r1))
                    matched_2.add((name, r2))
                    break

    return (only_1 - matched_1) | (only_2 - matched_2)


def _is_valid_kde_name(name: str) -> bool:
    """Return True only for plausible CIS section names."""
    s = str(name).strip()
    if not s.isascii():
        return False
    if len(s) > 60:
        return False
    if s.lower() in _INVALID_KDE_NAMES:
        return False
    if any(frag in s.lower() for frag in _INVALID_KDE_FRAGMENTS):
        return False
    return True


# ---------------------------------------------------------------------------
# Function 1: load_yaml_files
# ---------------------------------------------------------------------------
def load_yaml_files(yaml_path_1: str, yaml_path_2: str) -> tuple[dict, dict]:
    """
    Loads two YAML files produced by Task 1 and returns their contents
    as dictionaries.

    Args:
        yaml_path_1: Path to the first YAML file.
        yaml_path_2: Path to the second YAML file.

    Returns:
        A tuple of (kdes_1, kdes_2) dictionaries.

    Raises:
        FileNotFoundError: If either path does not exist.
        ValueError: If either file is not a .yaml/.yml file or cannot be parsed.
    """
    kdes = []
    for path in [yaml_path_1, yaml_path_2]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"YAML file not found: {path}")

        if not (path.lower().endswith(".yaml") or path.lower().endswith(".yml")):
            raise ValueError(f"Expected a YAML file, got: {path}")

        with open(path, "r", encoding="utf-8") as f:
            try:
                data = yaml.safe_load(f)
            except yaml.YAMLError as e:
                raise ValueError(f"Cannot parse YAML file '{path}': {e}")

        if not isinstance(data, dict):
            raise ValueError(f"YAML file does not contain a dictionary: {path}")

        kdes.append(data)

    return kdes[0], kdes[1]


# ---------------------------------------------------------------------------
# Function 2: compare_element_names
# ---------------------------------------------------------------------------
def compare_element_names(
    kdes_1: dict,
    kdes_2: dict,
    output_path: str = "outputs/diff_element_names.txt",
) -> str:
    """
    Identifies differences in element names between two KDE dictionaries.
    Writes a TEXT file listing the names that differ.

    Args:
        kdes_1: First KDE dictionary (from Task 1 YAML).
        kdes_2: Second KDE dictionary (from Task 1 YAML).
        output_path: Path for the output TEXT file.

    Returns:
        The path to the written output file.
    """
    names_1 = {v["name"] for v in kdes_1.values()
               if isinstance(v, dict) and "name" in v and _is_valid_kde_name(v["name"])}
    names_2 = {v["name"] for v in kdes_2.values()
               if isinstance(v, dict) and "name" in v and _is_valid_kde_name(v["name"])}

    # Symmetric difference: names in one but not the other
    diff = names_1.symmetric_difference(names_2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        if not diff:
            f.write("NO DIFFERENCES IN REGARDS TO ELEMENT NAMES\n")
        else:
            for name in sorted(diff):
                f.write(f"{name}\n")

    return output_path


# ---------------------------------------------------------------------------
# Function 3: compare_element_requirements
# ---------------------------------------------------------------------------
def compare_element_requirements(
    kdes_1: dict,
    kdes_2: dict,
    output_path: str = "outputs/diff_element_requirements.txt",
) -> str:
    """
    Identifies differences in element requirements between two KDE
    dictionaries. Writes a TEXT file with tuples of (NAME, REQUIREMENT)
    for each differing requirement.

    Args:
        kdes_1: First KDE dictionary (from Task 1 YAML).
        kdes_2: Second KDE dictionary (from Task 1 YAML).
        output_path: Path for the output TEXT file.

    Returns:
        The path to the written output file.
    """
    # Build sets of (name, requirement) tuples from each file
    pairs_1 = set()
    for v in kdes_1.values():
        if isinstance(v, dict) and "name" in v and "requirements" in v and _is_valid_kde_name(v["name"]):
            for req in v["requirements"]:
                pairs_1.add((v["name"], str(req)))

    pairs_2 = set()
    for v in kdes_2.values():
        if isinstance(v, dict) and "name" in v and "requirements" in v and _is_valid_kde_name(v["name"]):
            for req in v["requirements"]:
                pairs_2.add((v["name"], str(req)))

    # Fuzzy symmetric difference: collapses near-duplicate requirements
    diff = _fuzzy_dedup_pairs(pairs_1, pairs_2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        if not diff:
            f.write("NO DIFFERENCES IN REGARDS TO ELEMENT REQUIREMENTS\n")
        else:
            for name, req in sorted(diff):
                f.write(f"{name},{req}\n")

    return output_path
