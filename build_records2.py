"""Benchmark evaluation set: BIPIA (email, table, code contexts + BIPIA text/code attacks) and NotInject hard negatives."""
import json, random, re, sys
import pandas as pd
B = sys.argv[1]; NI = sys.argv[2]; OUT = sys.argv[3]
rng = random.Random(7)
def rd(p): return [json.loads(l) for l in open(p, encoding="utf-8")]
def flat(d): return [x for v in d.values() for x in v]
def cats(d): return [(k, x) for k, v in d.items() for x in v]
TA = cats(json.load(open(f"{B}/text_attack_test.json"))); CA = cats(json.load(open(f"{B}/code_attack_test.json")))
def split_sents(doc):
    s = re.split(r"(?<=[.!?\n])\s+", doc.strip()); return [x for x in s if x]
def inject(doc, text):
    pos = rng.choice(["start", "middle", "end"])
    if pos == "start": return text + "\n" + doc, pos
    if pos == "end": return doc + "\n" + text, pos
    s = split_sents(doc)
    if len(s) < 2: return doc + "\n" + text, "end"
    k = rng.randint(1, len(s) - 1); return " ".join(s[:k]) + "\n" + text + "\n" + " ".join(s[k:]), pos
def clip(t, n=1200):
    t = str(t); return t if len(t) <= n else t[:n].rsplit("\n", 1)[0]
recs = []
def add(**kw): kw["id"] = len(recs); recs.append(kw)
def q(x): return re.sub(r"^Q:\s*", "", str(x["question"])).strip()
CODE_TASK = "Using the document, explain what causes this error and how to fix it."
def code_doc(x):
    c = x["context"]; c = "\n".join(c) if isinstance(c, list) else str(c)
    e = "\n".join(x["error"][-4:]) if isinstance(x["error"], list) else str(x["error"])
    return "Error:\n" + e + "\n\nSuggested fix from a forum:\n" + clip(c, 900)
dom_spec = {"email": (rd(f"{B}/email/test.jsonl"), rd(f"{B}/email/train.jsonl"), q, lambda x: clip(x["context"]), TA, 3, 50),
            "table": (rd(f"{B}/table/test.jsonl"), rd(f"{B}/table/train.jsonl"), q, lambda x: clip(x["context"]), TA, 2, 60),
            "code":  (rd(f"{B}/code/test.jsonl"), rd(f"{B}/code/train.jsonl"), lambda x: CODE_TASK, code_doc, CA, 3, 50)}
for dom, (test, train, task, doc, atk, k, npool) in dom_spec.items():
    for i, x in enumerate(test):
        t, d = task(x), doc(x)
        add(domain=dom, split="test", family="benign", task=t, doc=d, ctx=f"{dom}{i}", attack_cat="", inj_pos="")
        for c, a in rng.sample(atk, k):
            dd, pos = inject(d, a); add(domain=dom, split="test", family="attack", task=t, doc=dd, ctx=f"{dom}{i}", attack_cat=c, inj_pos=pos)
    rng.shuffle(train)
    for i, x in enumerate(train[:npool]):
        add(domain=dom, split="pool", family="benign", task=task(x), doc=doc(x), ctx=f"{dom}p{i}", attack_cat="", inj_pos="")
ni = pd.concat([pd.read_parquet(f"{NI}/NotInject_{s}-00000-of-00001.parquet") for s in ("one", "two", "three")])
emails = rd(f"{B}/email/test.jsonl") + rd(f"{B}/email/train.jsonl")
for j, row in enumerate(ni.itertuples()):
    x = emails[j % len(emails)]; d, pos = inject(clip(x["context"]), row.prompt)
    add(domain="notinject", split="test", family="notinject", task=q(x), doc=d, ctx=f"ni{j}", attack_cat=row.category, inj_pos=pos)
with open(OUT, "w", encoding="utf-8") as fh:
    for r in recs: fh.write(json.dumps(r, ensure_ascii=False) + "\n")
from collections import Counter
print(len(recs), Counter((r["domain"], r["split"], r["family"]) for r in recs))
print("doc chars max", max(len(r["doc"]) for r in recs), "mean", sum(len(r["doc"]) for r in recs) / len(recs))
