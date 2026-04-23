# Sec-Unit — CIS Benchmark KDE Extraction Pipeline

Extracts Key Data Elements (KDEs) from CIS Benchmark security PDFs using a local LLM (Gemma-3-1B), then runs comparison and evaluation tasks across document pairs.

## Team Members

| Name | Auburn Email |
|---|---|
| Eli Musselwhite | esm0043@auburn.edu |
| Christopher Payne | cjp0099@auburn.edu |
| JT Nesbitt | jtn0035@auburn.edu |

## LLM

This project uses **[Gemma-3-1B-it](https://huggingface.co/google/gemma-3-1b-it)** (google/gemma-3-1b-it) for Task 1 KDE extraction.

## Requirements

- Python 3.12+
- CUDA-capable GPU strongly recommended (falls back to CPU automatically)
- For CUDA: install PyTorch with the appropriate CUDA index (see Setup)
- `kubescape` CLI on PATH (for Task 3)

## Setup

The easiest path is `./run.sh`, which creates the venv, installs `requirements.txt`,
and runs the full pipeline:

```bash
./run.sh                             # All 9 input combinations
./run.sh inputs/cis-r1.pdf inputs/cis-r2.pdf   # Single pair
./run.sh --build                     # Build PyInstaller binary only
```

Manual setup:

```bash
python -m venv comp5700-venv

# Linux/macOS
source comp5700-venv/bin/activate

# Windows
comp5700-venv\Scripts\activate

pip install -r requirements.txt
```

### PyTorch + CUDA

The default `requirements.txt` installs the CPU build of PyTorch. For GPU acceleration, reinstall torch with your CUDA version after the above step:

```bash
# CUDA 12.4 (most common for recent drivers)
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124

# CUDA 12.6
pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu126
```

Check your maximum supported CUDA version with `nvidia-smi` (top-right of the output). Use the closest `cu1XX` that does not exceed it.

### Inputs

Place your CIS Benchmark PDFs in the `inputs/` directory:

```
inputs/
  cis-r1.pdf
  cis-r2.pdf
  cis-r3.pdf
  cis-r4.pdf
```

## Usage

```bash
# All 9 predefined input combinations
python main.py --all

# Single pair
python main.py inputs/cis-r1.pdf inputs/cis-r2.pdf

# Custom input/output directories
python main.py --all --inputs-dir data/ --output-dir results/
```

Outputs are written to `outputs/` (gitignored).

### Running with the PyInstaller Binary

The binary is not committed (~2.4 GB — PyTorch + CUDA libs). Build it once:

```bash
./run.sh --build          # creates dist/sec-unit.exe (Windows) or dist/sec-unit (Linux/macOS)
```

Then run it:

```bash
./dist/sec-unit.exe --all
./dist/sec-unit.exe inputs/cis-r1.pdf inputs/cis-r2.pdf
```

## GPU & Batch Size

On startup the pipeline reads available VRAM and selects a batch size automatically (~1.5 GB per slot). All 6 prompts in a combo (3 prompt types × 2 documents) are sent to the model in one batched call, keeping the GPU fully utilised.

If an out-of-memory error occurs mid-run the batch size is halved automatically and the call is retried — no intervention needed. The reduced size is carried forward for all subsequent combos.

On CPU the batch size is fixed at 1.

## Running Tests

```bash
python -m unittest discover tests -v
```

## Prompt Strategies

Each document is processed with three prompt types defined in `PROMPT.md`:

| Strategy | Description |
|---|---|
| `zero_shot` | Direct extraction with no examples |
| `few_shot` | Two worked examples provided before the document |
| `chain_of_thought` | Step-by-step reasoning before final YAML output |

## Output Layout

For each input pair `(docA, docB)` the pipeline generates **six YAML files**
— one per `(document × prompt_type)` combination — because Task 1 requires all
three prompt strategies:

```
outputs/
  cis-r1-zero_shot-kdes.yaml
  cis-r1-few_shot-kdes.yaml
  cis-r1-chain_of_thought-kdes.yaml
  cis-r2-zero_shot-kdes.yaml
  cis-r2-few_shot-kdes.yaml
  cis-r2-chain_of_thought-kdes.yaml
```

Every filename includes the source document name, satisfying the spec's
"file names of the two YAML files include the names of the two input
documents" requirement; the prompt-type suffix disambiguates the three
strategies. Task 2 produces matching per-prompt-type diff files, and Task 3
merges the three prompt types via a consensus filter (a difference must
appear in ≥ 2 prompt types to drive Kubescape selection) before emitting
one CSV per combo under `outputs/<docA>_vs_<docB>/`.

## Document Chunking

CIS Benchmark PDFs are 200–230k characters — far beyond Gemma-3-1B's practical
context for a single call. Each document is split into overlapping ~20k-char
chunks (tunable via `SEC_UNIT_CHUNK_CHARS` / `SEC_UNIT_CHUNK_OVERLAP` env
vars), every chunk is prompted independently, and the resulting KDEs are
merged by element name — same-name elements have their requirements unioned
across chunks. This is what lets the pipeline surface KDEs from every
section of the benchmark, not just the first few pages.

## GitHub Actions Trigger

The assignment specifies running tests "every time a user types `git status`"
with `git commit` as a permitted alternative. Because GitHub Actions cannot
observe local `git status` calls, the workflow at `.github/workflows/tests.yml`
runs on `push` and `pull_request` — every commit pushed to any branch, and
every PR, triggers the full test suite.

## Project Structure

```
main.py          — Entry point; orchestrates the full pipeline
task1/           — KDE extraction (PDF parsing + LLM prompting)
task2/           — YAML comparator (element name & requirement diffs)
task3/           — Kubescape executor (control mapping, scanning, CSV)
tests/           — Test suite (unittest)
inputs/          — CIS Benchmark PDF inputs
PROMPT.md        — Full prompt text for all three strategies
requirements.txt — Python dependencies
```
