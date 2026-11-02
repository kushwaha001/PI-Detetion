# Hidden-state prompt-injection detectors under deployment shift

Code and results for a study of PIShield-style linear probes (a logistic-regression probe on one layer's hidden state at
the last prompt token) when the probe is trained in one domain and deployed in another. Six open models (Qwen2.5
0.5B / 1.5B / 3B / 7B, TinyLlama-1.1B, OLMo-2-1B), probes trained on Pune Metro documents with injected attacks, tested
on SQuAD, BIPIA (email, table, code) and NotInject.

## Findings

1. **Choosing the probe layer by in-domain validation fails under shift.** It picks early layers (2–7) that lose up to
   0.44 AUROC on BIPIA (mean regret 0.202). A label-free rule — first peak within 10% of the maximum of the benign
   intrinsic-dimension curve — has mean regret 0.021 and beats the validation pick on 5 of 6 models, with document-level
   bootstrap intervals clear of zero. It does not always find the best layer (7B: ~0.02 worse than mid-depth).
2. **Detection thresholds collapse in a new domain.** A 5% threshold set on the training domain flags 61–86% of benign
   documents elsewhere; re-setting it on 50 clean target documents brings this to 1–6%, no attack labels needed.
3. **That recalibration can be poisoned.** With a fraction e of the unlabeled pool contaminated, the benign FPR of a
   q-quantile threshold is only pinned to [(1−e−q)/(1−e), (1−q)/(1−e)]. 5% decoys drive every quantile threshold to 0%
   detection; a robust median/MAD transfer keeps 34–45% detection at 20% decoys on 3 of 4 models (fails on TinyLlama).
4. **Harmless trigger words remain the open problem.** Probes flag 54% (3B) and 63–79% (7B) of NotInject's benign
   documents after recalibration; scale does not help.

Supporting evidence: a white-box universal-suffix attack cuts a single-layer probe from 99% to 7% detection, and with
200 steps a two-layer (L14 OR L22) detector from 100% to 57%, consistent with prior work on obfuscated activations.

Full write-up: `repro/report/`. Every experiment, its question, scripts, jobs, outputs and result: `repro/EXPERIMENTS.md`.

## Layout

| Path | What |
|---|---|
| `layer_sweep_extract.py`, `layer_sweep_extract2.py`, `quant_loader.py`, `extract_v2.py` | All-layer hidden-state extraction (GPU; 4-bit loading for 3B/7B) |
| `layer_sweep_analyze.py` | Per-layer probes, TwoNN / Levina–Bickel intrinsic dimension with bootstraps |
| `cross_eval.py`, `ni_eval.py`, `reverse_eval.py`, `deberta_baseline.py` | Benchmark transfer, NotInject experiments, reverse direction, DeBERTa baseline (CPU) |
| `adaptive_attack.py`, `adaptive_attack2.py`, `adversarial_train.py` | White-box suffix attacks and adversarial training of the probes |
| `build_records2.py` | Builds the BIPIA + NotInject records |
| `repro/analysis/` | Layer-selection rules, bootstrap CIs, contaminated recalibration, benchmark summary |
| `repro/analysis_out/` | Outputs of those scripts as used in the write-up |
| `repro/jobs/` | Every job script in run order (`ORDER.txt` = runner log) |
| `repro/environment/` | `pip freeze` of both Python environments, GPU details |
| `repro/FROZEN_RULES.md` | The layer rule, fixed before the held-out models were run |
| `repro/MANIFEST.md5` | Checksums of all code |
| `overnight_runner.ps1` / `.bat` | The job-queue runner used for every GPU/CPU job |

## Reproducing

See `repro/README.md` for step-by-step commands. In short: run the extraction jobs (GPU) to write hidden states,
then the analysis jobs (CPU), then `repro/analysis/*.py`. The hidden-state arrays (several GB) and the source-domain
records are not in this repository; the extraction scripts regenerate the arrays from the models and records.

Environment used: Windows laptop, RTX 2050 (4 GB), torch 2.11.0+cu128, transformers 4.52.3, bitsandbytes 0.50.2;
analysis in Anaconda (numpy 1.26.4, scikit-learn). Seeds fixed (`numpy.random.default_rng(0)`, `torch.manual_seed(0)`).
