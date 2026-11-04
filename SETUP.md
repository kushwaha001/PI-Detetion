# Setup and running

Two Python environments were used. Python 3.11 (3.11.7 used).

| Environment | Used for | Needs GPU | Install |
|---|---|---|---|
| CPU | layer selection (ALS), correlations, confidence intervals, contamination, probes on saved states | No | `requirements-cpu.txt` |
| GPU | extracting hidden states from the models, white-box attacks, adversarial training | Yes (4 GB was enough; 3B/7B in 4-bit) | `requirements-gpu.txt` |

## Level 1 — rerun the ALS analysis from this repository alone (CPU, ~5 minutes)

The per-layer result tables (`results_sweep/**/*per_layer.json`, numbers only) are in the repository, so the
layer-selection results run without any model or dataset.

Windows (PowerShell):
```powershell
git clone https://github.com/kushwaha001/PI-Detetion.git
cd PI-Detetion
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements-cpu.txt
python repro\analysis\id_auroc_correlation.py      # ID vs detection, controlling for depth
python repro\analysis\als_checks.py forward         # ALS paper vs adapted rule, cutoff sensitivity, figure
```
Linux / macOS: same, with `python3.11 -m venv .venv` and `source .venv/bin/activate`.

Outputs: `repro/analysis_out/id_auroc_correlation.md`, `repro/analysis_out/als_checks/sensitivity.md`, `als_figure.png`.
They should match the committed files (tested from a fresh clone; only the last floating-point digit can differ).

## Level 2 — the full pipeline (GPU + data)

### 1. Environment
```powershell
py -3.11 -m venv D:\ldg-venv
D:\ldg-venv\Scripts\Activate.ps1
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-gpu.txt
```
Any recent CUDA build of torch should work; the results used `torch 2.11.0+cu128` (driver 592.00).

### 2. Models (Hugging Face originals) into `D:\ldg-models\`
`Qwen2.5-0.5B-Instruct`, `Qwen2.5-1.5B-Instruct`, `Qwen2.5-3B-Instruct`, `Qwen2.5-7B-Instruct`,
`TinyLlama-1.1B-Chat-v1.0`, `OLMo-2-0425-1B-Instruct`, and the baseline `protectai/deberta-v3-base-prompt-injection-v2`.
Example: `huggingface-cli download Qwen/Qwen2.5-1.5B-Instruct --local-dir D:\ldg-models\Qwen2.5-1.5B-Instruct`

### 3. Data
| Data | Where the code expects it |
|---|---|
| Pune Metro document chunks | `D:\PuneMetroAssistant\output\chunks.pkl` (`extract_v2.py`, `load_metro`) — not public; any set of plain-prose documents of 300–800 characters can stand in |
| SQuAD paragraphs | `C:\Users\ASUS\trustworthy-ai-week1\data\squad_mini.json` (`load_squad`; a JSON with a `corpus` list of paragraphs) |
| BIPIA | BIPIA benchmark folder (github.com/microsoft/BIPIA), passed to `build_records2.py` |
| NotInject | `D:\ldg-data\NotInject` (`huggingface-cli download leolee99/NotInject --repo-type dataset`) |

Change these paths at the top of `extract_v2.py` (and in the job scripts) if your folders differ.

### 4. Run
Each job in `repro/jobs/` is a short PowerShell script that sets environment variables and calls one Python script;
`repro/jobs/ORDER.txt` lists the order they ran in. Either run a job directly, or copy jobs into `results_sweep\queue\`
and start `overnight_runner.bat`. In order:

1. `extract_v2.py` builds `cache_v2\records_full.jsonl` (records) for the source domain.
2. `layer_sweep_extract.py` (GPU) writes all-layer states to `D:\ldg-sweep\<model>` — jobs `*_x_*` / `*extract*`.
3. `layer_sweep_analyze.py` (CPU) — per-layer probes and ID curves → `results_sweep\<model>\per_layer.json`.
4. `repro\analysis\select_math.py` — ALS and the other rules → `repro\analysis_out\selection\<model>.json`.
5. Benchmarks: `build_records2.py`, `layer_sweep_extract2.py` (GPU), `cross_eval.py`, `ni_eval.py`, `reverse_eval.py`.
6. Statistics: `repro\analysis\ci.py`, `bench_summary.py`, `contamination.py`, `als_checks.py`, `id_auroc_correlation.py`.
7. Attacks: `adaptive_attack.py`, `adaptive_attack2.py`, `adversarial_train.py` (GPU).

Model-specific settings: 3B and 7B need `QUANT=4bit` (7B also `DEVICE_MAP=layers_gpu`); OLMo-2 needs `DTYPE=bf16`.
Exact commands and settings for every step are in the job files; `repro/EXPERIMENTS.md` maps each result to its job.
