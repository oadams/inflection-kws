""" Interface to the inflection generation tools """

import logging
from pathlib import Path

HYPS_DIR = Path(f"../../G2PHypotheses/")

def load_hypotheses(iso_code, k=None, method="DTL", pos_sets=["nouns"],
                    hyps_dir=HYPS_DIR):
    """ Loads hypothesized inflections

        method is a string that indicates what method was used to generate the
        hypotheses. The default is DTL, which uses Garrett Nicolai's DTL
        variation.

        k is the number of inflections per (lemma, bundle) pair to use.

        pos_sets specifies what pos sets to explore. The default is nouns.
    """

    # NOTE Commented out the below block because I'm worried about Congo
    # Swahili not being similar enough to Coastal Swahili, so it should just
    # fail loudly.
    # if iso_code == "swc":
    #     # Then use *swh* *verbs* instead.
    #     hyps_path = Path(f"/export/a14/yarowsky-lab/gnicolai/G2PHypotheses/swh.verbs.out")

    if method == "DTL":
        suffix = "out"
    elif method == "ensemble":
        suffix = "Ens"
    elif method == "RNN":
        suffix = "RNN"
    else:
        raise ValueError(f"Invalid method {method}")


    hyps = {}

    for pos_set in pos_sets:
        hyps_path = hyps_dir / f"{iso_code}.{pos_set}.{suffix}"
        logging.info(f"Loading inflection hypotheses from {hyps_path}...")
        with open(hyps_path) as f:
            for line in f:
                fields = line.split("\t")
                if len(fields) != 5:
                    # Then it's probably an empty line between bundles
                    continue
                lemma, bundle = fields[0].split("+")
                inflection_hyp = fields[1]
                if inflection_hyp == "":
                    # We don't use empty inflections.
                    continue

                if lemma in hyps:
                    if bundle in hyps[lemma]:
                        hyps[lemma][bundle].append(inflection_hyp)
                    else:
                        hyps[lemma][bundle] = [inflection_hyp]
                else:
                    hyps[lemma] = {bundle: [inflection_hyp]}

    if k:
        for lemma in hyps:
            for bundle in hyps[lemma]:
                hyps[lemma][bundle] = hyps[lemma][bundle][:k]
                assert len(hyps[lemma][bundle]) <= k

    return hyps
