# Label-free layer selection for probe-based prompt-injection detection

Reproducibility package for the experiments that apply ALS-style layer selection (intrinsic-dimension
curves, from hallucination detection) to PIShield-style linear probes for indirect prompt injection (PI).
Results write-up: "Probe-based PI detection under domain shift — results" (Claude Docs). A Markdown
export is in `report/` when available.

Everything below runs from the project root `C:\Users\ASUS\latent-drift-guard` (call it ROOT).

## Layout

| Where | What |
|---|---|
| `ROOT\*.py` | GPU and CPU pipeline scripts (the code of record; md5 in `repro\MANIFEST.md5`) |
| `ROOT\cache_v2\records_full.jsonl` | Source-domain records: metro (train/cal/test) and SQuAD (test), families benign / A / B / C / D |
| `ROOT\results_sweep\bench\records2.jsonl` | Benchmark records: BIPIA email/table/code (test + clean pool) and NotInject (1,199 records) |
| `ROOT\results_sweep\<model>\` | Per-model layer sweep: `per_layer.json`, `probe_scores.npz`, `report.md`, plot |
| `ROOT\results_sweep\bench\<model>\` | Cross-domain evaluation on BIPIA / NotInject: `bench_per_layer.json`, `bench_scores.npz` |
| `ROOT\results_sweep\reverse\<model>\` | Reverse direction (train on BIPIA): `reverse_per_layer.json` |
| `ROOT\results_sweep\attack\`, `attack2\` | Adaptive white-box suffix attacks |
| `ROOT\results_sweep\FROZEN_RULES.md` | Layer-selection rule frozen before the held-out models were run |
| `D:\ldg-sweep\...` | Large hidden-state memmaps (`final.f16`, `done.npy`, `meta.json`), one folder per model; `bench\<model>` for benchmarks |
| `D:\ldg-models\`, `ROOT\models\` | Model weights (Hugging Face originals) |
| `repro\jobs\` | Every job script that was run, in run order (`ORDER.txt` = provenance from the runner log) |
| `repro\analysis\` | CPU analysis scripts that ran outside the laptop (layer-selection rules, bootstrap CIs, contamination, benchmark summary) |
| `repro\analysis_out\` | Their outputs as used in the write-up |
| `repro\environment\` | `pip freeze` of both Python environments |

## Environments

- GPU env `D:\ldg-venv` (torch 2.11.0+cu128, transformers 4.52.3, bitsandbytes 0.50.2, accelerate 1.14.0, numpy 1.26.4; full list in `environment\gpu_env_ldg-venv.txt`). Used for extraction and attacks.
- CPU env `C:\Users\ASUS\anaconda3` (numpy, scikit-learn, matplotlib). Used for probe training and analysis.
- Hardware: 4 GB laptop GPU. Models up to 1.5B run in fp16/bf16; 3B and 7B in 4-bit NF4 (`QUANT=4bit`, 7B also `DEVICE_MAP=layers_gpu`).
- Seeds: analysis uses `numpy.random.default_rng(0)`; probes are deterministic logistic regression (C = 0.2).

## How to reproduce

Jobs are PowerShell scripts that set environment variables and call one Python script. `overnight_runner.bat`
runs every `*.ps1` in `results_sweep\queue\` oldest first, logs to `queue\done\`, and stops on a `STOP` file or after 10 h.
To rerun a step, copy its job from `repro\jobs\` into `results_sweep\queue\` and start the runner, or run it directly in PowerShell.

1. **Layer sweep, source domain** (per model): `layer_sweep_extract.py` (GPU) writes all-layer last-prompt-token states to `D:\ldg-sweep\<model>`; `layer_sweep_analyze.py` (CPU) trains per-layer probes, computes TwoNN / Levina–Bickel intrinsic dimension with bootstraps, and writes `results_sweep\<model>\`. Jobs `*_x_<model>` and `*_an_<model>`.
2. **Benchmarks**: `build_records2.py` builds `records2.jsonl`; `layer_sweep_extract2.py` (GPU) extracts benchmark states; `cross_eval.py` (CPU) applies metro-trained probes to BIPIA/NotInject (std and hard-negative variants). Jobs `*_x2_<model>`, `*_ce_<model>`.
3. **DeBERTa baseline**: `deberta_baseline.py` (job `job26_deberta`).
4. **Reverse direction**: `reverse_eval.py` trains on BIPIA email+table and tests on BIPIA code, metro and SQuAD (jobs `job40`–`job41`, `job60`–`job63`).
5. **Adaptive attacks**: `adaptive_attack.py` (one layer, `ATTACK_LAYER`) and `adaptive_attack2.py` (`ATTACK_LAYERS="14,22"`, `STEPS`, `NSUF`, `NCAND`).
6. **Analysis** (CPU, any machine with numpy + scikit-learn; reads `cache_v2` and `results_sweep`):

   ```
   cd ROOT
   # layer-selection rules (strict/smoothed FEPoID, sig(z), knee, near-top, PIShield val_acc, mid, last), one model at a time
   set OUT=repro\analysis_out\selection\qwen2.5-3b.json
   python repro\analysis\select_math.py cache_v2\records_full.jsonl results_sweep\qwen2.5-3b
   # benchmark regret table across models
   python repro\analysis\bench_summary.py
   # document-level bootstrap CIs (B=500 used for 7B/3B, 200 for the rest; SKIP=model,... to skip)
   set B=200
   python repro\analysis\ci.py
   # recalibration from contaminated traffic; argument maps model -> [probe_scores.npz, layer] (the ID-rule layer)
   python repro\analysis\contamination.py "{\"qwen2.5-1.5b\": [\"results_sweep/qwen2.5-1.5b/probe_scores.npz\", 17]}"
   ```
   `ci.py` expects selection files named as in its `SEL` map (`tinyllama.json`, `olmo.json` for those two models).
   Checked on Nov 2: rerunning `select_math.py` on Qwen2.5-3B from this folder gives the same layer picks as the archived file.

## Known issues

- 7B run is trimmed (4 train tasks, 2 test tasks on metro/SQuAD) because 4-bit inference with offloading ran at 0.07 records/s.
- The 200-step / 25-token joint attack (job67) stalled after 20 steps on Nov 2; job71 (`NSUF=20`, `NCAND=24`) completed instead.
- Bootstrap CIs for 1.5B, 0.5B, TinyLlama and OLMo-2 use 200 resamples (the cloud machine restarted during the 500-resample run).
- OLMo-2 must use `DTYPE=bf16` (fp16 overflows).

See `EXPERIMENTS.md` for every experiment, the question behind it, the exact scripts/jobs and the result.
