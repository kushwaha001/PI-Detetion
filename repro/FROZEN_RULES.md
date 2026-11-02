# Layer-selection rules frozen before TinyLlama-1.1B and OLMo-2-1B results exist
Frozen: 2026-10-28 07:35 IST (TinyLlama extraction at ~1600/10400; OLMo-2 not started).
Developed on: Qwen2.5-1.5B (all rules) and Qwen2.5-0.5B (neartop was proposed after seeing it).

Input: TwoNN intrinsic dimension of last-prompt-token (or drift-delta) states of 1,600 benign metro-train prompts,
10 bootstrap subsamples per layer; m_l = bootstrap mean, se_l = bootstrap std. No attack data is used.

Primary rule  NEARTOP(eps=0.10): first local maximum l of m with m_l >= 0.9 * max_j m_j.
Secondary     SIG(z=1.645, w=5): first local maximum p of m such that max_{p<j<=p+5} (m_j - m_p)/sqrt(se_p^2+se_j^2) < 1.645.
Also reported KNEE(eps=0.10): first l with m_l >= 0.9 * max m;  SMOOTH: ALS FEPoID on 3-layer moving average;
              STRICT: ALS FEPoID; PIShield validation accuracy; fixed 0.5 depth; last layer.
Score: regret = max_l AUROC_SQuAD(l) - AUROC_SQuAD(selected l), mean over attack families A/B/C, leave-one-family-out probe.

Results at freeze time (regret, final / delta):
  Qwen2.5-1.5B: NEARTOP 0.002/0.002, SIG 0.002/0.002, KNEE 0.011/0.002, PIShield 0.115/0.115, fixed-0.5 0.011/0.007
  Qwen2.5-0.5B: NEARTOP 0.003/0.000, SIG 0.063/0.051, KNEE 0.003/0.000, PIShield 0.132/0.129, fixed-0.5 0.012/0.018
