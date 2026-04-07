"""
Task 3: Executor Module
========================
Four functions for reading Task 2 diff files, mapping differences to
Kubescape controls, running Kubescape scans, and generating CSV reports.
"""

import os
import re
import subprocess
import zipfile
import pandas as pd


# ---------------------------------------------------------------------------
# Mapping from KDE names / keywords to Kubescape control IDs
# Based on https://kubescape.io/docs/controls/
# ---------------------------------------------------------------------------
KEYWORD_TO_CONTROLS = {
    "logging": ["C-0067"],
    "audit": ["C-0067", "C-0068"],
    "kubelet": ["C-0057", "C-0058", "C-0059"],
    "anonymous auth": ["C-0058"],
    "authentication": ["C-0058"],
    "authorization": ["C-0057"],
    "pod security": ["C-0044", "C-0045"],
    "privilege": ["C-0016", "C-0017", "C-0057"],
    "network": ["C-0049", "C-0050"],
    "network policy": ["C-0049", "C-0050"],
    "secret": ["C-0012", "C-0034"],
    "rbac": ["C-0035", "C-0036", "C-0037"],
    "role": ["C-0035", "C-0036", "C-0037"],
    "tls": ["C-0032"],
    "encryption": ["C-0032", "C-0034"],
    "container": ["C-0016", "C-0017", "C-0044"],
    "image": ["C-0030"],
    "namespace": ["C-0038"],
    "resource limit": ["C-0009"],
    "limit": ["C-0009"],
    "host": ["C-0038", "C-0041", "C-0044"],
    "capability": ["C-0016", "C-0046"],
    "seccomp": ["C-0055"],
    "service account": ["C-0035", "C-0036"],
    "etcd": ["C-0032", "C-0067"],
    "api server": ["C-0057", "C-0058", "C-0067"],
    "control plane": ["C-0057", "C-0058", "C-0067"],
    "scheduler": ["C-0057"],
    "configuration": ["C-0034"],
    "permission": ["C-0057"],
    "file": ["C-0034"],
    "ownership": ["C-0034"],
}


# ---------------------------------------------------------------------------
# Function 1: load_diff_files
# ---------------------------------------------------------------------------
def load_diff_files(
    names_diff_path: str, reqs_diff_path: str
) -> tuple[str, str]:
    """
    Loads the two TEXT files produced by Task 2.

    Args:
        names_diff_path: Path to the element-names diff file.
        reqs_diff_path: Path to the requirements diff file.

    Returns:
        A tuple of (names_content, reqs_content) strings.

    Raises:
        FileNotFoundError: If either file does not exist.
        ValueError: If either file is empty.
    """
    contents = []
    for path in [names_diff_path, reqs_diff_path]:
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Diff file not found: {path}")

        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        if not content.strip():
            raise ValueError(f"Diff file is empty: {path}")

        contents.append(content)

    return contents[0], contents[1]


# ---------------------------------------------------------------------------
# Function 2: determine_controls
# ---------------------------------------------------------------------------
def determine_controls(
    names_content: str,
    reqs_content: str,
    output_path: str = "outputs/controls.txt",
) -> str:
    """
    Determines whether the diff files indicate differences. If at least
    one difference exists, maps the differences to Kubescape control IDs.
    Otherwise reports 'NO DIFFERENCES FOUND'.

    Args:
        names_content: Content of the element-names diff file.
        reqs_content: Content of the requirements diff file.
        output_path: Path for the output TEXT file.

    Returns:
        The path to the written output file.
    """
    no_name_diff = "NO DIFFERENCES IN REGARDS TO ELEMENT NAMES" in names_content
    no_req_diff = "NO DIFFERENCES IN REGARDS TO ELEMENT REQUIREMENTS" in reqs_content

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if no_name_diff and no_req_diff:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write("NO DIFFERENCES FOUND\n")
        return output_path

    # Collect all text from diffs for keyword matching
    combined_text = ""
    if not no_name_diff:
        combined_text += names_content + "\n"
    if not no_req_diff:
        combined_text += reqs_content + "\n"

    combined_lower = combined_text.lower()

    # Map keywords to controls
    matched_controls = set()
    for keyword, controls in KEYWORD_TO_CONTROLS.items():
        if keyword in combined_lower:
            matched_controls.update(controls)

    # If no keywords matched but differences exist, include a broad default set
    if not matched_controls:
        matched_controls = {"C-0016", "C-0034", "C-0044", "C-0057"}

    with open(output_path, "w") as f:
        for control in sorted(matched_controls):
            f.write(f"{control}\n")

    return output_path


# ---------------------------------------------------------------------------
# Function 3: run_kubescape
# ---------------------------------------------------------------------------
def run_kubescape(
    controls_path: str,
    yamls_path: str = "project-yamls.zip",
    kubescape_cmd: str = "kubescape",
) -> pd.DataFrame:
    """
    Executes Kubescape on the provided YAML files based on the controls
    file content. Returns scan results as a pandas DataFrame.

    Args:
        controls_path: Path to the controls TEXT file from determine_controls().
        yamls_path: Path to project-yamls.zip or a directory of YAML files.
        kubescape_cmd: Command/path for the kubescape binary.

    Returns:
        A pandas DataFrame with scan results.

    Raises:
        FileNotFoundError: If controls file or yamls path doesn't exist.
        RuntimeError: If kubescape execution fails.
    """
    if not os.path.isfile(controls_path):
        raise FileNotFoundError(f"Controls file not found: {controls_path}")

    # If the yamls_path is a zip, extract it to a temp directory
    scan_target = yamls_path
    extracted_dir = None
    if yamls_path.endswith(".zip"):
        if not os.path.isfile(yamls_path):
            raise FileNotFoundError(f"YAML archive not found: {yamls_path}")
        extracted_dir = yamls_path.replace(".zip", "")
        os.makedirs(extracted_dir, exist_ok=True)
        with zipfile.ZipFile(yamls_path, "r") as zf:
            zf.extractall(extracted_dir)
        scan_target = extracted_dir

    with open(controls_path, "r", encoding="utf-8") as f:
        controls_content = f.read().strip()

    output_dir = os.path.dirname(os.path.abspath(controls_path))
    # Kubescape requires a relative path for --output on some platforms;
    # we write to CWD then move to the per-combo output directory.
    results_json_tmp = "kubescape_results.json"
    results_json = os.path.join(output_dir, "kubescape_results.json")

    # Build the kubescape command
    cmd = [kubescape_cmd, "scan"]

    if "NO DIFFERENCES FOUND" in controls_content:
        # Run with all controls
        cmd += [scan_target, "--format", "json", "--output", results_json_tmp]
    else:
        # Run only on the specific controls
        control_ids = [
            line.strip()
            for line in controls_content.split("\n")
            if line.strip()
        ]
        controls_arg = ",".join(control_ids)
        cmd += [
            "control",
            controls_arg,
            scan_target,
            "--format",
            "json",
            "--output",
            results_json_tmp,
        ]

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode not in (0, 1):
        # kubescape returns 1 when controls fail (expected), other codes are errors
        raise RuntimeError(
            f"Kubescape failed (exit {result.returncode}):\n{result.stderr}"
        )

    # Parse JSON results into a DataFrame
    import json
    import shutil

    if not os.path.isfile(results_json_tmp):
        raise RuntimeError("Kubescape did not produce a results file.")

    shutil.move(results_json_tmp, results_json)

    with open(results_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    resource_map = {
        r["resourceID"]: r.get("source", {}).get("relativePath", "")
        for r in data.get("resources", [])
        if "resourceID" in r
    }

    rows = []
    for resource_result in data.get("results", []):
        file_path = resource_map.get(resource_result.get("resourceID", ""), "")

        for control_result in resource_result.get("controls", []):
            control_name = control_result.get("name", "")
            control_id = control_result.get("controlID", "")
            severity = control_result.get("severity", {})
            if isinstance(severity, dict):
                severity = severity.get("scoreFactor", "")
            status = control_result.get("status", {})
            if isinstance(status, dict):
                status = status.get("status", "")

            rows.append(
                {
                    "FilePath": file_path,
                    "Severity": severity,
                    "Control name": f"{control_id} - {control_name}" if control_id else control_name,
                    "Status": status,
                }
            )

    if not rows:
        # Build from summary if detailed results aren't available
        for ctrl in data.get("summaryDetails", {}).get("controls", {}).values():
            rows.append(
                {
                    "FilePath": scan_target,
                    "Severity": ctrl.get("scoreFactor", ""),
                    "Control name": ctrl.get("name", ""),
                    "Failed resources": ctrl.get("ResourceCounters", {}).get(
                        "failedResources", 0
                    ),
                    "All Resources": ctrl.get("ResourceCounters", {}).get(
                        "allResources", 0
                    ),
                    "Compliance score": ctrl.get("complianceScore", 0),
                }
            )
        return pd.DataFrame(rows)

    df = pd.DataFrame(rows)

    # Aggregate to get Failed/All resource counts and compliance per control
    if "Status" in df.columns:
        summary = (
            df.groupby(["FilePath", "Severity", "Control name"])
            .agg(
                Failed_resources=("Status", lambda x: (x == "failed").sum()),
                All_Resources=("Status", "count"),
            )
            .reset_index()
        )
        summary["Compliance score"] = round(
            (1 - summary["Failed_resources"] / summary["All_Resources"].replace(0, 1))
            * 100,
            2,
        )
        summary = summary.rename(
            columns={
                "Failed_resources": "Failed resources",
                "All_Resources": "All Resources",
            }
        )
        return summary

    return df


# ---------------------------------------------------------------------------
# Function 4: generate_csv
# ---------------------------------------------------------------------------
def generate_csv(
    df: pd.DataFrame, output_path: str = "outputs/scan_results.csv"
) -> str:
    """
    Generates a CSV file from the Kubescape scan results DataFrame.

    Args:
        df: DataFrame with scan results.
        output_path: Path for the output CSV file.

    Returns:
        The path to the written CSV file.
    """
    required_columns = [
        "FilePath",
        "Severity",
        "Control name",
        "Failed resources",
        "All Resources",
        "Compliance score",
    ]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # Ensure all required columns exist, filling missing ones with defaults
    for col in required_columns:
        if col not in df.columns:
            df[col] = ""

    df[required_columns].to_csv(output_path, index=False)
    return output_path
