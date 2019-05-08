""" Interface to the inflection generation tools """

from pathlib import Path

DTL_HYPS_DIR = Path(f"/../../G2PHypotheses/")

def load_hypotheses(iso_code, method="DTL",
                               pos_sets=["nouns"]):
    """ Loads hypothesized inflections

        method is a string that indicates what method was used to generate the
        hypotheses. The default is DTL, which uses Garrett Nicolai's DTL
        variation.

        pos_sets specifies what pos sets to explore. The default is nouns.
    """

    # NOTE Commented out the below block because I'm worried about Congo
    # Swahili not being similar enough to Coastal Swahili, so it should just
    # fail loudly.
    # if iso_code == "swc":
    #     # Then use *swh* *verbs* instead.
    #     hyps_path = Path(f"/export/a14/yarowsky-lab/gnicolai/G2PHypotheses/swh.verbs.out")

    if method == "DTL":
        hyps_dir = DTL_HYPS_DIR

    hyps = {}

    for pos_set in pos_sets:
        hyps_path = hyps_dir / f"{iso_code}.{pos_set}.out"
        with open(hyps_path) as f:
            for line in f:
                fields = line.split("\t")
                lemma, bundle = fields[0].split("+")
                inflection_hyp = fields[1]

                if lemma in hyps:
                    if bundle in hyps[lemma]:
                        hyps[lemma][bundle].append(inflection_hyp)
                    else:
                        hyps[lemma][bundle] = [inflection_hyp]
                else:
                    hyps[lemma] = {bundle: [inflection_hyp]}

    return hyps
