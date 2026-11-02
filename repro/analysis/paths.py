"""Shared paths for the analysis scripts. Run them from anywhere; set LDG_ROOT if the project is not two levels up."""
import os
from pathlib import Path
ROOT = Path(os.environ.get("LDG_ROOT", Path(__file__).resolve().parents[2]))
SWEEP = ROOT / "results_sweep"                         # per-model results written by the laptop pipeline
REC1 = ROOT / "cache_v2" / "records_full.jsonl"        # metro/SQuAD records (source domain)
REC2 = SWEEP / "bench" / "records2.jsonl"              # BIPIA + NotInject records
OUT = ROOT / "repro" / "analysis_out"; OUT.mkdir(parents=True, exist_ok=True)
SEL = OUT / "selection"; SEL.mkdir(parents=True, exist_ok=True)
