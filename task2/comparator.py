"""
Task 2: Comparator Module
==========================
Three functions for loading KDE YAML files from Task 1 and identifying
differences in element names and requirements between two documents.
"""

import os
import re
from collections import defaultdict
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
    s = _LEADING_VERB_RE.sub("", s)
    s = " ".join(s.lower().split())
    return s


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


def _extract_name_to_reqs(kdes: dict) -> dict[str, list[str]]:
    """Flatten a KDE dict to {name: [req, ...]}, filtering invalid names."""
    out: dict[str, list[str]] = defaultdict(list)
    for v in kdes.values():
        if not isinstance(v, dict):
            continue
        name = v.get("name")
        reqs = v.get("requirements", [])
        if not name or not _is_valid_kde_name(name):
            continue
        if isinstance(reqs, str):
            reqs = [reqs]
        for r in reqs:
            out[name].append(str(r))
    return dict(out)


def _fuzzy_diff_reqs(
    reqs_a: list[str], reqs_b: list[str], threshold: float = 0.85
) -> tuple[list[str], list[str]]:
    """
    Symmetric difference of two requirement lists, collapsing fuzzy matches.
    Returns (only_in_a, only_in_b).
    """
    set_a = set(reqs_a)
    set_b = set(reqs_b)
    only_a = list(set_a - set_b)
    only_b = list(set_b - set_a)

    if not only_a or not only_b:
        return only_a, only_b

    matched_a: set[str] = set()
    matched_b: set[str] = set()
    for r1 in only_a:
        n1 = _normalize_req(r1)
        for r2 in only_b:
            if r2 in matched_b:
                continue
            n2 = _normalize_req(r2)
            if n1 == n2 or SequenceMatcher(None, n1, n2).ratio() >= threshold:
                matched_a.add(r1)
                matched_b.add(r2)
                break

    return [r for r in only_a if r not in matched_a], [r for r in only_b if r not in matched_b]


# ---------------------------------------------------------------------------
# Function 1: load_yaml_files
# ---------------------------------------------------------------------------
def load_yaml_files(yaml_path_1: str, yaml_path_2: str) -> tuple[dict, dict]:
    """
    Loads two YAML files produced by Task 1 and returns their contents
    as dictionaries.
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
    Writes a TEXT file listing the names that differ, or the 'NO DIFFERENCES'
    sentinel if there are none.
    """
    names_1 = {v["name"] for v in kdes_1.values()
               if isinstance(v, dict) and "name" in v and _is_valid_kde_name(v["name"])}
    names_2 = {v["name"] for v in kdes_2.values()
               if isinstance(v, dict) and "name" in v and _is_valid_kde_name(v["name"])}

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
    file_1_name: str = "file1",
    file_2_name: str = "file2",
    output_path: str = "outputs/diff_element_requirements.txt",
) -> str:
    """
    Identifies differences in both (i) element names and (ii) requirements
    between two KDE dictionaries, written as CSV tuples per the spec:

        NAME,ABSENT-IN-<filename>,PRESENT-IN-<filename>,NA   # name-only diff
        NAME,ABSENT-IN-<filename>,PRESENT-IN-<filename>,REQ  # requirement diff

    Args:
        kdes_1: First KDE dictionary (from Task 1 YAML).
        kdes_2: Second KDE dictionary (from Task 1 YAML).
        file_1_name: Display name of the first file (appears in markers).
        file_2_name: Display name of the second file (appears in markers).
        output_path: Path for the output TEXT file.
    """
    map_1 = _extract_name_to_reqs(kdes_1)
    map_2 = _extract_name_to_reqs(kdes_2)

    lines: list[str] = []

    # (i) Name-only differences — KDE in one file, absent in the other
    for name in sorted(set(map_1) - set(map_2)):
        lines.append(f"{name},ABSENT-IN-{file_2_name},PRESENT-IN-{file_1_name},NA")
    for name in sorted(set(map_2) - set(map_1)):
        lines.append(f"{name},ABSENT-IN-{file_1_name},PRESENT-IN-{file_2_name},NA")

    # (ii) Requirement differences — KDE present in both, but req differs
    for name in sorted(set(map_1) & set(map_2)):
        only_1, only_2 = _fuzzy_diff_reqs(map_1[name], map_2[name])
        for req in sorted(only_1):
            lines.append(f"{name},ABSENT-IN-{file_2_name},PRESENT-IN-{file_1_name},{req}")
        for req in sorted(only_2):
            lines.append(f"{name},ABSENT-IN-{file_1_name},PRESENT-IN-{file_2_name},{req}")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        if not lines:
            f.write("NO DIFFERENCES IN REGARDS TO ELEMENT REQUIREMENTS\n")
        else:
            for line in lines:
                f.write(line + "\n")

    return output_path
