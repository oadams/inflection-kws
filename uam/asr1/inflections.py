""" Interface to the inflection generation tools """

from pathlib import Path

# TODO Generalize this beyond nouns.
DTL_HYPS_DIR = Path(f"/export/a14/yarowsky-lab/gnicolai/G2PHypotheses/")

def load_dtl_hypotheses(iso_code, dtl_hyps_dir=DTL_HYPS_DIR):
    """ Loads hypothesized inflections produced using Garrett Nicolai's DTL
    variation.
    """

    # NOTE Commented out the below block because I'm worried about Congo
    # Swahili not being similar enough to Coastal Swahili.
    # if iso_code == "swc":
    #     # Then use *swh* *verbs* instead.
    #     hyps_path = Path(f"/export/a14/yarowsky-lab/gnicolai/G2PHypotheses/swh.verbs.out")

    hyps_path = DTL_HYPS_DIR / f"{iso_code}.nouns.out"

    hyps = {}
    with open(hyps_path) as f:
        for line in f:
            fields = line.split("\t")
            bundle, lemma = fields[0].split("+")
            inflection_hyp = fields[1]

            if lemma in hyps:
                if bundle in hyps[lemma]:
                    hyps[lemma][bundle].append(inflection_hyp)
                else:
                    hyps[lemma][bundle] = [inflection_hyp]
            else:
                hyps[lemma] = {bundle: [inflection_hyp]}

    return hyps
