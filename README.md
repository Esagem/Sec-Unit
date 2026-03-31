# Sec-Unit — CIS Benchmark KDE Extraction Pipeline

Extracts Key Data Elements (KDEs) from CIS Benchmark security PDFs using a local LLM (Gemma-3-1B), then runs comparison and evaluation tasks across document pairs.

## Setup

```bash
python -m venv comp5700-venv
source comp5700-venv/bin/activate   # Windows: comp5700-venv\Scripts\activate
pip install -r requirements.txt
```

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
# Single pair
python main.py inputs/cis-r1.pdf inputs/cis-r2.pdf

# Task 1 only
python main.py --task1 inputs/cis-r1.pdf inputs/cis-r2.pdf

# All 9 predefined input combinations
python main.py --all

# Custom input/output directories
python main.py --all --inputs-dir data/ --output-dir results/
```

Outputs are written to `outputs/` (gitignored).

## Prompt Strategies

Each document is processed with three prompt types defined in `PROMPT.md`:

| Strategy | Description |
|---|---|
| `zero_shot` | Direct extraction with no examples |
| `few_shot` | Two worked examples provided before the document |
| `chain_of_thought` | Step-by-step reasoning before final YAML output |

## Project Structure

```
main.py          — Entry point; orchestrates the full pipeline
task1/           — KDE extraction (PDF parsing + LLM prompting)
task2/           — (in progress)
task3/           — (in progress)
tests/           — Pytest test suite
inputs/          — CIS Benchmark PDF inputs
PROMPT.md        — Full prompt text for all three strategies
requirements.txt — Python dependencies
```

## Requirements

- Python 3.12+
- CUDA-capable GPU recommended (falls back to CPU)
