"""
Task 2: Comparator Module
==========================
Three functions for loading KDE YAML files from Task 1 and identifying
differences in element names and requirements between two documents.
"""

import os
import yaml


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

        with open(path, "r") as f:
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
    names_1 = {v["name"] for v in kdes_1.values() if isinstance(v, dict) and "name" in v}
    names_2 = {v["name"] for v in kdes_2.values() if isinstance(v, dict) and "name" in v}

    # Symmetric difference: names in one but not the other
    diff = names_1.symmetric_difference(names_2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
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
        if isinstance(v, dict) and "name" in v and "requirements" in v:
            for req in v["requirements"]:
                pairs_1.add((v["name"], str(req)))

    pairs_2 = set()
    for v in kdes_2.values():
        if isinstance(v, dict) and "name" in v and "requirements" in v:
            for req in v["requirements"]:
                pairs_2.add((v["name"], str(req)))

    # Symmetric difference: pairs present in one but not the other
    diff = pairs_1.symmetric_difference(pairs_2)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with open(output_path, "w") as f:
        if not diff:
            f.write("NO DIFFERENCES IN REGARDS TO ELEMENT REQUIREMENTS\n")
        else:
            for name, req in sorted(diff):
                f.write(f"{name},{req}\n")

    return output_path
