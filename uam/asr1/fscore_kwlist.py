from pathlib import Path

k=1
babel_code = "206"

"""
exp_prefix = "new-tur"
from lxml import etree

ref_kwlist_path = Path(f"kwlists/_{exp_prefix}_k={k}/{babel_code}.kwlist.xml")
hyp_kwlist_path = Path(f"kwlists/_{exp_prefix}_kwset-spurious_k={k}/{babel_code}.kwlist.xml")

with open(ref_kwlist_path) as f:
    ref_root = etree.fromstring(f.read())
with open(hyp_kwlist_path) as f:
    hyp_root = etree.fromstring(f.read())

print(ref_root.tag)
ref = dict()
for child in ref_root:
    ref
    print(child[0].text)

for kw in hyp_root:
    for kwtext in kw:
        print(kwtext.text)
"""

import kws_eval

for k in range(1,10):
    ref_paradigms = kws_eval.create_eval_paradigms(babel_code, "ensemble",
                                                   k=k,
                                                   write_to_fn=True,
                                                   kwset_spurious=False,
                                                   kwset_affix=f"k-{k}")
    hyp_paradigms = kws_eval.create_eval_paradigms(babel_code, "ensemble",
                                                   k=k,
                                                   write_to_fn=True,
                                                   kwset_spurious=True,
                                                   kwset_affix=f"kwset-spurious_k-{k}")
    tp = 0
    fp = 0
    fn = 0
    print(len(ref_paradigms))
    print(len(hyp_paradigms))
    for lemma in ref_paradigms:
        for inflection in hyp_paradigms[lemma]:
            if inflection in ref_paradigms[lemma]:
                tp += 1
            else:
                fp += 1
        for inflection in ref_paradigms[lemma]:
            if not inflection in hyp_paradigms[lemma]:
                fn += 1

    print(f"tp: {tp}")
    print(f"fp: {fp}")
    print(f"fn: {fn}")
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    for lemma in hyp_paradigms:
        print(f"Size of first infleciton set: {len(hyp_paradigms[lemma])}")
        break
    print(f"@k={k}, precision: {precision}", flush=True)
    print(f"@k={k}, recall: {recall}", flush=True)
