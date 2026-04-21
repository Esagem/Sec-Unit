"""
Benchmark script — measures runtime of each task in the Sec-Unit pipeline.

Task 1 costs (PDF parse, model load, LLM inference) are measured on one pair
of documents; Task 2 and Task 3 costs are measured across the full set of
Task-1 output YAMLs so the numbers reflect realistic end-to-end timing.

Usage:
    python benchmark.py [--skip-llm]   # Skip Task 1 (model load + inference)
"""

import argparse
import os
import sys
import time
from contextlib import contextmanager

# Silence transformers chatter when we do load it
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

INPUTS_DIR = "inputs"
OUTPUT_DIR = "outputs"
BENCH_DIR = "bench_outputs"
DOCS = ["cis-r1", "cis-r2", "cis-r3", "cis-r4"]
PROMPT_TYPES = ["zero_shot", "few_shot", "chain_of_thought"]
COMBOS = [
    ("cis-r1", "cis-r1"), ("cis-r1", "cis-r2"), ("cis-r1", "cis-r3"),
    ("cis-r1", "cis-r4"), ("cis-r2", "cis-r2"), ("cis-r2", "cis-r3"),
    ("cis-r2", "cis-r4"), ("cis-r3", "cis-r3"), ("cis-r3", "cis-r4"),
]


@contextmanager
def timer(label: str, results: dict):
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    results[label] = elapsed
    print(f"  {label:<42} {elapsed:8.3f}s")


def bench_task1_parse_only(results: dict):
    """Measure PDF load + prompt construction cost (no LLM)."""
    from task1.extractor import load_documents, build_messages

    pdf1 = os.path.join(INPUTS_DIR, "cis-r1.pdf")
    pdf2 = os.path.join(INPUTS_DIR, "cis-r2.pdf")

    print("\n[Task 1 — PDF parse + prompt build, no LLM]")
    with timer("load_documents (cis-r1, cis-r2)", results):
        t1, t2 = load_documents(pdf1, pdf2)

    with timer("build_messages × 6 (3 prompts × 2 docs)", results):
        for pt in PROMPT_TYPES:
            for text in (t1, t2):
                build_messages(text, pt)


def bench_task1_llm(results: dict):
    """Measure model load + inference on one document pair."""
    import torch
    from transformers import pipeline, GenerationConfig
    import transformers
    transformers.logging.set_verbosity_error()

    from task1.extractor import load_documents, build_messages, save_kde_result

    pdf1 = os.path.join(INPUTS_DIR, "cis-r1.pdf")
    pdf2 = os.path.join(INPUTS_DIR, "cis-r2.pdf")

    print("\n[Task 1 — Gemma-3-1B load + inference]")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  device: {device}")

    with timer("pipeline load (Gemma-3-1B)", results):
        pipe = pipeline(
            "text-generation",
            model="google/gemma-3-1b-it",
            device=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        pipe.model.generation_config = GenerationConfig(
            do_sample=False, max_new_tokens=1024
        )

    t1, t2 = load_documents(pdf1, pdf2)
    messages_list = []
    job_meta = []
    for pt in PROMPT_TYPES:
        for doc_name, text in [("cis-r1", t1), ("cis-r2", t2)]:
            m, p = build_messages(text, pt)
            messages_list.append(m)
            job_meta.append((doc_name, pt, p))

    os.makedirs(BENCH_DIR, exist_ok=True)
    with timer("inference × 6 prompts (batched)", results):
        outputs = pipe(messages_list, batch_size=6 if device == "cuda" else 1)

    with timer("YAML serialization × 6", results):
        for i, (doc, pt, prompt_text) in enumerate(job_meta):
            raw = outputs[i][0]["generated_text"][-1]["content"]
            save_kde_result(
                raw_response=raw, prompt_text=prompt_text, prompt_type=pt,
                doc_name=f"{doc}-{pt}", output_dir=BENCH_DIR,
            )


def bench_task2(results: dict):
    """Run comparator across all 9 combos × 3 prompt types."""
    from task2.comparator import (
        load_yaml_files, compare_element_names, compare_element_requirements,
    )

    print("\n[Task 2 — Comparator on all 9 combos × 3 prompt types = 27 comparisons]")
    os.makedirs(BENCH_DIR, exist_ok=True)
    pairs = []
    for d1, d2 in COMBOS:
        for pt in PROMPT_TYPES:
            y1 = os.path.join(OUTPUT_DIR, f"{d1}-{pt}-kdes.yaml")
            y2 = os.path.join(OUTPUT_DIR, f"{d2}-{pt}-kdes.yaml")
            if os.path.exists(y1) and os.path.exists(y2):
                pairs.append((y1, y2, d1, d2, pt))

    if not pairs:
        print("  (no YAMLs found — run Task 1 first)")
        return

    with timer(f"load_yaml_files × {len(pairs)}", results):
        loaded = [(load_yaml_files(y1, y2), d1, d2, pt) for y1, y2, d1, d2, pt in pairs]

    with timer(f"compare_element_names × {len(pairs)}", results):
        for (k1, k2), d1, d2, pt in loaded:
            compare_element_names(
                k1, k2,
                output_path=os.path.join(BENCH_DIR, f"diff_names_{d1}_vs_{d2}_{pt}.txt"),
            )

    with timer(f"compare_element_requirements × {len(pairs)}", results):
        for (k1, k2), d1, d2, pt in loaded:
            compare_element_requirements(
                k1, k2,
                file_1_name=f"{d1}-{pt}-kdes",
                file_2_name=f"{d2}-{pt}-kdes",
                output_path=os.path.join(BENCH_DIR, f"diff_reqs_{d1}_vs_{d2}_{pt}.txt"),
            )


def bench_task3(results: dict):
    """Task 3 non-kubescape pieces (controls mapping + CSV generation)."""
    from task3.executor import load_diff_files, determine_controls, generate_csv
    import pandas as pd

    print("\n[Task 3 — Control mapping + CSV generation (no kubescape)]")
    os.makedirs(BENCH_DIR, exist_ok=True)

    nf = os.path.join(OUTPUT_DIR, "diff_names_cis-r1-zero_shot-kdes_vs_cis-r2-zero_shot-kdes.txt")
    rf = os.path.join(OUTPUT_DIR, "diff_reqs_cis-r1-zero_shot-kdes_vs_cis-r2-zero_shot-kdes.txt")
    if not (os.path.exists(nf) and os.path.exists(rf)):
        print("  (no diff files found — run Task 2 first)")
        return

    with timer("load_diff_files", results):
        nc, rc = load_diff_files(nf, rf)

    with timer("determine_controls", results):
        determine_controls(nc, rc, output_path=os.path.join(BENCH_DIR, "controls.txt"))

    sample_df = pd.DataFrame([{
        "FilePath": "deployment.yaml", "Severity": 8,
        "Control name": "C-0057", "Failed resources": 1,
        "All Resources": 2, "Compliance score": 50.0,
    }] * 100)
    with timer("generate_csv (100 rows)", results):
        generate_csv(sample_df, output_path=os.path.join(BENCH_DIR, "scan_results.csv"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-llm", action="store_true", help="Skip Gemma load + inference")
    args = parser.parse_args()

    results: dict[str, float] = {}
    print("=" * 60)
    print("Sec-Unit Pipeline Benchmark")
    print("=" * 60)

    bench_task1_parse_only(results)
    if not args.skip_llm:
        try:
            bench_task1_llm(results)
        except Exception as e:
            print(f"  (LLM bench skipped: {e})")
    bench_task2(results)
    bench_task3(results)

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    total = sum(results.values())
    for label, elapsed in results.items():
        pct = 100 * elapsed / total if total else 0
        print(f"  {label:<42} {elapsed:8.3f}s  ({pct:5.1f}%)")
    print(f"  {'TOTAL':<42} {total:8.3f}s")


if __name__ == "__main__":
    main()
