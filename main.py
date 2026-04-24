"""
main.py — Project Entry Point
===============================
Runs the full pipeline: Task 1 → Task 2 → Task 3.
Can be invoked with two PDF paths or run all 9 combinations.

Usage:
    python main.py <pdf1> <pdf2>           # Single pair
    python main.py --all                   # All 9 combinations
    python main.py --task1 <pdf1> <pdf2>   # Task 1 only

AI Usage Statement:
We JT Nesbitt, Eli Musselwhite, and Christo Payne would like to acknowledge the use of ChatGPT and Claude,
language models developed by OpenAI and Anthropic, in preparation of this assignment. They were used in
the creation of code specifically in the design and implementation of various
components of this project, including debugging code, refining logic, improving
structure, and clarifying concepts related to the CIS Benchmark pipline, document
processing, and Kubescape integration.

AI tools were also used to help interpret error messages, suggest optimizations
and provide guidance on overall system design. 

Final design was reviewed, tested, and refined by the authors. All work 
submitted represents our understanding of the project. 


"""

import argparse
import os
import sys
import warnings
import logging
warnings.filterwarnings("ignore")
logging.disable(logging.CRITICAL)

from rich.progress import (
    Progress,
    SpinnerColumn,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
    TaskProgressColumn,
)
from rich.console import Console
from rich.panel import Panel

console = Console()

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
_CONSENSUS_MIN = 2  # A diff line must appear in at least this many prompt types


def get_model_pipeline():
    """Load the Gemma-3-1B pipeline. Called once, shared across runs."""
    import transformers
    from transformers import pipeline, GenerationConfig
    import torch

    transformers.logging.set_verbosity_error()

    with console.status("[bold cyan]Loading Gemma-3-1B model…[/bold cyan]", spinner="dots2"):
        pipe = pipeline(
            "text-generation",
            model="google/gemma-3-1b-it",
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16,
        )
        pipe.model.generation_config = GenerationConfig(do_sample=False, max_new_tokens=3072)
        pipe.model = torch.compile(pipe.model)

    console.print("[bold green]✓[/bold green] Model loaded.")
    return pipe


def _detect_batch_size() -> int:
    """
    Estimate a safe starting batch size based on free VRAM.
    Falls back to 1 on CPU or if VRAM cannot be queried.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 1
        free_bytes, _ = torch.cuda.mem_get_info()
        free_gb = free_bytes / 1024 ** 3
        # ~1.5 GB per sequence is a conservative estimate for Gemma-3-1B
        # with 2048-token KV cache at fp16
        batch = max(1, int(free_gb // 2.0))
        console.print(f"[dim]GPU free VRAM: {free_gb:.1f} GB → starting batch size: {batch}[/dim]")
        return batch
    except Exception:
        return 1


def _run_with_fallback(pipe, all_messages: list, batch_size: int) -> tuple[list, int]:
    """
    Run the pipeline with the given batch_size, halving on OOM until batch_size=1.
    Returns (outputs, final_batch_size) so the caller can persist the new size.
    """
    import torch
    while batch_size >= 1:
        try:
            outputs = pipe(all_messages, batch_size=batch_size)
            return outputs, batch_size
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            batch_size = max(1, batch_size // 2)
            console.print(f"[yellow]⚠ OOM — reducing batch size to {batch_size}[/yellow]")
            if batch_size == 1:
                outputs = pipe(all_messages, batch_size=1)
                return outputs, 1
    return pipe(all_messages, batch_size=1), 1


def run_task1(pdf1_path: str, pdf2_path: str, pipe, output_dir: str = "outputs", pbar=None,
              batch_size: int = 1):
    """
    Run Task 1 for a single pair of PDFs across all 3 prompt types.
    Batches all 6 prompts into a single pipeline call for GPU efficiency.
    Returns a list of result dicts.
    """
    from task1.extractor import (
        load_documents,
        build_messages,
        save_kde_result,
        collect_llm_output,
    )

    # Load documents
    text_1, text_2 = load_documents(pdf1_path, pdf2_path)
    doc1_name = os.path.splitext(os.path.basename(pdf1_path))[0]
    doc2_name = os.path.splitext(os.path.basename(pdf2_path))[0]

    # Each (prompt, doc) job may expand into multiple chunks. We flatten all
    # chunks across all 6 jobs into a single batched pipeline call, then
    # regroup the responses per (prompt, doc) before parsing + merging.
    chunk_jobs: list[tuple[list, str]] = []          # (messages, prompt_text) per chunk
    group_indices: list[tuple[str, str, str, list[int]]] = []
    # (prompt_type, doc_name, representative_prompt_text, [indices into chunk_jobs])

    for prompt_type in PROMPT_TYPES:
        for doc_name, text in [(doc1_name, text_1), (doc2_name, text_2)]:
            chunk_msgs = build_messages(text, prompt_type)
            indices = []
            for messages, prompt_text in chunk_msgs:
                indices.append(len(chunk_jobs))
                chunk_jobs.append((messages, prompt_text))
            group_indices.append((prompt_type, doc_name, chunk_msgs[0][1], indices))

    if pbar is not None:
        progress, task = pbar
        progress.update(task, description=f"[bold cyan]{doc1_name} vs {doc2_name}[/bold cyan]")

    # Batched pipeline call with automatic OOM fallback
    all_messages = [j[0] for j in chunk_jobs]
    outputs, batch_size = _run_with_fallback(pipe, all_messages, batch_size)

    # Regroup: collect each group's per-chunk responses, merge into one YAML
    all_results = []
    for prompt_type, doc_name, prompt_text, indices in group_indices:
        raw_responses = [
            outputs[i][0]["generated_text"][-1]["content"] for i in indices
        ]
        result = save_kde_result(
            raw_response=raw_responses,
            prompt_text=prompt_text,
            prompt_type=prompt_type,
            doc_name=f"{doc_name}-{prompt_type}",
            output_dir=output_dir,
        )
        all_results.append(result)
        if pbar is not None:
            progress.advance(task)

    # Collect all LLM outputs into a single text file
    combo_name = f"{doc1_name}_vs_{doc2_name}"
    output_txt = os.path.join(output_dir, f"llm_output_{combo_name}.txt")
    collect_llm_output(all_results, output_path=output_txt)

    return all_results, batch_size


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
        file_1_name=base1,
        file_2_name=base2,
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

    console.print(Panel.fit(
        "[bold cyan]CIS Benchmark Security Pipeline[/bold cyan]\n"
        "[dim]Gemma-3-1B · KDE Extraction · Kubescape Analysis[/dim]",
        border_style="cyan",
    ))

    # Load model once
    pipe = get_model_pipeline()
    batch_size = _detect_batch_size()

    progress = Progress(
        SpinnerColumn(spinner_name="dots2", style="bold magenta"),
        TextColumn("[bold blue]{task.description}"),
        BarColumn(bar_width=40, style="cyan", complete_style="bold green", finished_style="bold green"),
        TaskProgressColumn(),
        TextColumn("[dim]·[/dim]"),
        TimeElapsedColumn(),
        TextColumn("[dim]eta[/dim]"),
        TimeRemainingColumn(),
        console=console,
        transient=False,
    )

    if args.all:
        total_calls = len(INPUT_COMBOS) * len(PROMPT_TYPES) * 2
        with progress:
            task = progress.add_task("[cyan]Extracting KDEs…", total=total_calls)
            for pdf1_name, pdf2_name in INPUT_COMBOS:
                pdf1 = os.path.join(args.inputs_dir, pdf1_name)
                pdf2 = os.path.join(args.inputs_dir, pdf2_name)
                _, batch_size = run_task1(
                    pdf1, pdf2, pipe,
                    output_dir=args.output_dir,
                    pbar=(progress, task),
                    batch_size=batch_size,
                )
        console.print("[bold green]✓ Task 1 complete.[/bold green]")

        if not args.task1:
            console.print("\n[bold cyan]Running Task 2: KDE Comparison…[/bold cyan]")
            for pdf1_name, pdf2_name in INPUT_COMBOS:
                doc1 = os.path.splitext(pdf1_name)[0]
                doc2 = os.path.splitext(pdf2_name)[0]
                for prompt_type in PROMPT_TYPES:
                    yaml1 = os.path.join(args.output_dir, f"{doc1}-{prompt_type}-kdes.yaml")
                    yaml2 = os.path.join(args.output_dir, f"{doc2}-{prompt_type}-kdes.yaml")
                    if os.path.exists(yaml1) and os.path.exists(yaml2):
                        run_task2(yaml1, yaml2, output_dir=args.output_dir)
            console.print("[bold green]✓ Task 2 complete.[/bold green]")

            console.print("\n[bold cyan]Running Task 3: Kubescape Analysis…[/bold cyan]")
            for pdf1_name, pdf2_name in INPUT_COMBOS:
                doc1 = os.path.splitext(pdf1_name)[0]
                doc2 = os.path.splitext(pdf2_name)[0]
                combo_out = os.path.join(args.output_dir, f"{doc1}_vs_{doc2}")
                os.makedirs(combo_out, exist_ok=True)
                from collections import Counter
                name_counts: Counter[str] = Counter()
                req_counts: Counter[str] = Counter()
                for pt in PROMPT_TYPES:
                    combo_pt = f"{doc1}-{pt}-kdes_vs_{doc2}-{pt}-kdes"
                    nf = os.path.join(args.output_dir, f"diff_names_{combo_pt}.txt")
                    rf = os.path.join(args.output_dir, f"diff_reqs_{combo_pt}.txt")
                    for path, counter in [(nf, name_counts), (rf, req_counts)]:
                        if os.path.exists(path):
                            content = open(path, encoding="utf-8").read().strip()
                            if "NO DIFFERENCES" not in content:
                                for line in content.split('\n'):
                                    if line.strip():
                                        counter[line.strip()] += 1
                # Only keep lines that at least _CONSENSUS_MIN prompt types agree on
                merged_names = {l for l, c in name_counts.items() if c >= _CONSENSUS_MIN}
                merged_reqs = {l for l, c in req_counts.items() if c >= _CONSENSUS_MIN}
                merged_names_path = os.path.join(combo_out, "merged_diff_names.txt")
                merged_reqs_path = os.path.join(combo_out, "merged_diff_reqs.txt")
                with open(merged_names_path, "w", encoding="utf-8") as f:
                    f.write('\n'.join(sorted(merged_names)) + '\n' if merged_names else "NO DIFFERENCES IN REGARDS TO ELEMENT NAMES\n")
                with open(merged_reqs_path, "w", encoding="utf-8") as f:
                    f.write('\n'.join(sorted(merged_reqs)) + '\n' if merged_reqs else "NO DIFFERENCES IN REGARDS TO ELEMENT REQUIREMENTS\n")
                if not merged_names and not merged_reqs:
                    console.print(f"[dim]  ⏭ {doc1} vs {doc2}: no differences — skipping Kubescape[/dim]")
                    continue
                try:
                    run_task3(merged_names_path, merged_reqs_path, output_dir=combo_out)
                except Exception as e:
                    console.print(f"[yellow]⚠ Task 3 skipped for {doc1} vs {doc2}: {e}[/yellow]")
            console.print("[bold green]✓ Task 3 complete.[/bold green]")

        console.print("[bold green]✓ All runs complete.[/bold green]")

    elif args.pdf1 and args.pdf2:
        total_calls = len(PROMPT_TYPES) * 2
        with progress:
            task = progress.add_task("[cyan]Extracting KDEs…", total=total_calls)
            run_task1(
                args.pdf1, args.pdf2, pipe,
                output_dir=args.output_dir,
                pbar=(progress, task),
                batch_size=batch_size,
            )
        console.print("[bold green]✓ Task 1 complete.[/bold green]")

        if not args.task1:
            doc1 = os.path.splitext(os.path.basename(args.pdf1))[0]
            doc2 = os.path.splitext(os.path.basename(args.pdf2))[0]

            console.print("\n[bold cyan]Running Task 2: KDE Comparison…[/bold cyan]")
            for prompt_type in PROMPT_TYPES:
                yaml1 = os.path.join(args.output_dir, f"{doc1}-{prompt_type}-kdes.yaml")
                yaml2 = os.path.join(args.output_dir, f"{doc2}-{prompt_type}-kdes.yaml")
                if os.path.exists(yaml1) and os.path.exists(yaml2):
                    run_task2(yaml1, yaml2, output_dir=args.output_dir)
            console.print("[bold green]✓ Task 2 complete.[/bold green]")

            console.print("\n[bold cyan]Running Task 3: Kubescape Analysis…[/bold cyan]")
            combo_out = os.path.join(args.output_dir, f"{doc1}_vs_{doc2}")
            os.makedirs(combo_out, exist_ok=True)
            from collections import Counter
            name_counts: Counter[str] = Counter()
            req_counts: Counter[str] = Counter()
            for pt in PROMPT_TYPES:
                combo_pt = f"{doc1}-{pt}-kdes_vs_{doc2}-{pt}-kdes"
                nf = os.path.join(args.output_dir, f"diff_names_{combo_pt}.txt")
                rf = os.path.join(args.output_dir, f"diff_reqs_{combo_pt}.txt")
                for path, counter in [(nf, name_counts), (rf, req_counts)]:
                    if os.path.exists(path):
                        content = open(path, encoding="utf-8").read().strip()
                        if "NO DIFFERENCES" not in content:
                            for line in content.split('\n'):
                                if line.strip():
                                    counter[line.strip()] += 1
            merged_names = {l for l, c in name_counts.items() if c >= _CONSENSUS_MIN}
            merged_reqs = {l for l, c in req_counts.items() if c >= _CONSENSUS_MIN}
            merged_names_path = os.path.join(combo_out, "merged_diff_names.txt")
            merged_reqs_path = os.path.join(combo_out, "merged_diff_reqs.txt")
            with open(merged_names_path, "w", encoding="utf-8") as f:
                f.write('\n'.join(sorted(merged_names)) + '\n' if merged_names else "NO DIFFERENCES IN REGARDS TO ELEMENT NAMES\n")
            with open(merged_reqs_path, "w", encoding="utf-8") as f:
                f.write('\n'.join(sorted(merged_reqs)) + '\n' if merged_reqs else "NO DIFFERENCES IN REGARDS TO ELEMENT REQUIREMENTS\n")
            if not merged_names and not merged_reqs:
                console.print("[dim]  ⏭ No differences — skipping Kubescape[/dim]")
            else:
                try:
                    run_task3(merged_names_path, merged_reqs_path, output_dir=combo_out)
                except Exception as e:
                    console.print(f"[yellow]⚠ Task 3 skipped: {e}[/yellow]")
            console.print("[bold green]✓ Done.[/bold green]")

    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
