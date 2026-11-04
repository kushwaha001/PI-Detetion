# ALS in the reverse direction

Probe trained on BIPIA email+table (reverse_eval.py). ALS reads the TwoNN curve of benign BIPIA email+table prompts (test and pool), 10 subsamples of 80%. Regret against the best layer for metro and for SQuAD.

| Model | Best layer, metro | Best layer, SQuAD | ALS paper | ALS adapted | Forward ALS layer | Regret metro: paper / adapted / forward | Regret SQuAD: paper / adapted / forward |
|---|---|---|---|---|---|---|---|
| qwen2.5-0.5b | L6 | L7 | L12 | L15 | L14 | 0.173 / 0.049 / 0.092 | 0.182 / 0.069 / 0.110 |
| qwen2.5-1.5b | L15 | L15 | L7 | L21 | L17 | 0.244 / 0.149 / 0.040 | 0.165 / 0.142 / 0.034 |
| qwen2.5-3b | L26 | L25 | L10 | L30 | L26 | 0.191 / 0.166 / 0.000 | 0.078 / 0.128 / 0.021 |
| qwen2.5-7b | L8 | L16 | L27 | L27 | L19 | 0.230 / 0.230 / 0.185 | 0.347 / 0.347 / 0.211 |
| tinyllama-1.1b | L7 | L7 | L19 | L19 | L13 | 0.138 / 0.138 / 0.090 | 0.041 / 0.041 / 0.057 |
| olmo2-1b | L3 | L3 | L11 | L11 | L11 | 0.097 / 0.097 / 0.097 | 0.043 / 0.043 / 0.043 |

Mean regret_metro: paper 0.179, adapted 0.138, forward_als 0.084

Mean regret_squad: paper 0.143, adapted 0.128, forward_als 0.079