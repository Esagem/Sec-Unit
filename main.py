"""
main.py — Project Entry Point
===============================
Runs the full pipeline: Task 1 → Task 2 → Task 3.
Can be invoked with two PDF paths or run all 9 combinations.

Usage:
    python main.py <pdf1> <pdf2>           # Single pair
    python main.py --all                   # All 9 combinations
    python main.py --task1 <pdf1> <pdf2>   # Task 1 only
"""

import argparse
import os
import sys
import itertools

# ---------------------------------------------------------------------------
# Input combinations as defined by the project spec
# ---------------------------------------------------------------------------
INPUT_COMBOS = [
    ("cis-r1.pdf", "cis-r1.pdf"),  # Input-1
    ("cis-r1.pdf", "cis-r2.pdf"),  # Input-2
    ("cis-r1.pdf", "cis-r3.pdf"),  # Input-3
    ("cis-r1.pdf", "cis-r4.pdf"),  # Input-4
    ("cis-r2.pdf", "cis-r2.pdf"),  # Input-5
    ("cis-r2.pdf", "cis-r3.pdf"),  # Input-6
    ("cis-r2.pdf", "cis-r4.pdf"),  # Input-7
    ("cis-r3.pdf", "cis-r3.pdf"),  # Input-8
    ("cis-r3.pdf", "cis-r4.pdf"),  # Input-9
]

PROMPT_TYPES = ["zero_shot", "few_shot", "chain_of_thought"]


def get_model_pipeline():
    """Load the Gemma-3-1B pipeline. Called once, shared across runs."""
    from transformers import pipeline
    import torch

    print("[*] Loading Gemma-3-1B model (this may take a minute)...")
    pipe = pipeline(
        "text-generation",
        model="google/gemma-3-1b-it",
        device="cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.float16,
    )
    print("[+] Model loaded.")
    return pipe


def run_task1(pdf1_path: str, pdf2_path: str, pipe, output_dir: str = "outputs"):
    """
    Run Task 1 for a single pair of PDFs across all 3 prompt types.
    Returns a list of result dicts.
    """
    from task1.extractor import (
        load_documents,
        extract_kdes,
        collect_llm_output,
    )

    print(f"\n{'='*60}")
    print(f"Task 1: {os.path.basename(pdf1_path)} + {os.path.basename(pdf2_path)}")
    print(f"{'='*60}")

    # Load documents
    text_1, text_2 = load_documents(pdf1_path, pdf2_path)
    doc1_name = os.path.splitext(os.path.basename(pdf1_path))[0]
    doc2_name = os.path.splitext(os.path.basename(pdf2_path))[0]

    all_results = []

    # Run each prompt type on each document
    for prompt_type in PROMPT_TYPES:
        print(f"\n  [{prompt_type}] Processing {doc1_name}...")
        result_1 = extract_kdes(
            document_text=text_1,
            prompt_type=prompt_type,
            pipe=pipe,
            doc_name=f"{doc1_name}-{prompt_type}",
            output_dir=output_dir,
        )
        all_results.append(result_1)

        print(f"  [{prompt_type}] Processing {doc2_name}...")
        result_2 = extract_kdes(
            document_text=text_2,
            prompt_type=prompt_type,
            pipe=pipe,
            doc_name=f"{doc2_name}-{prompt_type}",
            output_dir=output_dir,
        )
        all_results.append(result_2)

    # Collect all LLM outputs into a single text file
    combo_name = f"{doc1_name}_vs_{doc2_name}"
    output_txt = os.path.join(output_dir, f"llm_output_{combo_name}.txt")
    collect_llm_output(all_results, output_path=output_txt)
    print(f"\n  [+] LLM output log: {output_txt}")

    return all_results


def run_task2(
    yaml_path_1: str, yaml_path_2: str, output_dir: str = "outputs"
) -> dict:
    """
    Run Task 2 for a pair of YAML files produced by Task 1.
    Compares element names and requirements, writes two TEXT files.
    Returns a dict with paths to the output files.
    """
    from task2.comparator import (
        load_yaml_files,
        compare_element_names,
        compare_element_requirements,
    )

    print(f"\n{'='*60}")
    print(f"Task 2: {os.path.basename(yaml_path_1)} vs {os.path.basename(yaml_path_2)}")
    print(f"{'='*60}")

    kdes_1, kdes_2 = load_yaml_files(yaml_path_1, yaml_path_2)

    base1 = os.path.splitext(os.path.basename(yaml_path_1))[0]
    base2 = os.path.splitext(os.path.basename(yaml_path_2))[0]
    combo = f"{base1}_vs_{base2}"

    names_path = compare_element_names(
        kdes_1,
        kdes_2,
        output_path=os.path.join(output_dir, f"diff_names_{combo}.txt"),
    )
    print(f"  [+] Element name differences: {names_path}")

    reqs_path = compare_element_requirements(
        kdes_1,
        kdes_2,
        output_path=os.path.join(output_dir, f"diff_reqs_{combo}.txt"),
    )
    print(f"  [+] Requirement differences:  {reqs_path}")

    return {"names_diff_path": names_path, "reqs_diff_path": reqs_path}


def run_task3(
    names_diff_path: str,
    reqs_diff_path: str,
    yamls_path: str = "project-yamls.zip",
    output_dir: str = "outputs",
) -> str:
    """
    Run Task 3: determine controls, execute Kubescape, generate CSV.
    Returns the path to the output CSV file.
    """
    from task3.executor import (
        load_diff_files,
        determine_controls,
        run_kubescape,
        generate_csv,
    )

    print(f"\n{'='*60}")
    print("Task 3: Executor")
    print(f"{'='*60}")

    names_content, reqs_content = load_diff_files(names_diff_path, reqs_diff_path)

    controls_path = determine_controls(
        names_content,
        reqs_content,
        output_path=os.path.join(output_dir, "controls.txt"),
    )
    print(f"  [+] Controls file: {controls_path}")

    df = run_kubescape(controls_path, yamls_path=yamls_path)
    print(f"  [+] Kubescape scan complete: {len(df)} rows")

    csv_path = generate_csv(df, output_path=os.path.join(output_dir, "scan_results.csv"))
    print(f"  [+] CSV report: {csv_path}")

    return csv_path


def main():
    parser = argparse.ArgumentParser(description="CIS Benchmark Security Pipeline")
    parser.add_argument("pdf1", nargs="?", help="Path to first PDF")
    parser.add_argument("pdf2", nargs="?", help="Path to second PDF")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all 9 input combinations",
    )
    parser.add_argument(
        "--task1",
        action="store_true",
        help="Run Task 1 only",
    )
    parser.add_argument(
        "--inputs-dir",
        default="inputs",
        help="Directory containing the CIS PDF files",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for all output files",
    )

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # Load model once
    pipe = get_model_pipeline()

    if args.all:
        # Run all 9 combinations
        for pdf1_name, pdf2_name in INPUT_COMBOS:
            pdf1 = os.path.join(args.inputs_dir, pdf1_name)
            pdf2 = os.path.join(args.inputs_dir, pdf2_name)
            run_task1(pdf1, pdf2, pipe, output_dir=args.output_dir)
    elif args.pdf1 and args.pdf2:
        run_task1(args.pdf1, args.pdf2, pipe, output_dir=args.output_dir)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
