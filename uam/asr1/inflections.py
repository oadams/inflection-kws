""" Interface to the inflection generation tools """

import logging
from pathlib import Path

# NOTE this G2PHypotheses directory actually contains inflections and should be
# renamed.
HYPS_DIR = Path(f"../../G2PHypotheses/")

def load_lemmas_only(iso_code, pos_sets, unimorph_dir=Path("../../raw/unimorph")):
    inflections = dict()
    lang_path = unimorph_dir / f"{iso_code}/{iso_code}"
    lemma = None
    with open(lang_path) as f:
        for line in f:
            sp = line.split("\t")
            if len(sp) == 3:
                lemma, inflection, bundle = sp
                if "nouns" in pos_sets:
                    if bundle.split(";")[0] != "N":
                        continue
                # Just throw the lemma in as the only instance.
                inflections[lemma] = {bundle: [lemma]}
    return inflections

def load_unimorph_inflections(iso_code, pos_sets, unimorph_dir=Path("../../raw/unimorph")):
    """ Given an ISO 639-3 language code, returns a mapping from lemmas of that
        language to list of tuples of <inflection, unimorph bundle>.
    """

    logging.info(f"Loading inflections for {iso_code}")

    inflections = dict()
    lang_path = unimorph_dir / f"{iso_code}/{iso_code}"
    lemma = None
    with open(lang_path) as f:
        for line in f:
            sp = line.split("\t")
            if len(sp) == 3:
                lemma, inflection, bundle = sp
                if "nouns" in pos_sets:
                    if bundle.split(";")[0] != "N":
                        continue
                if lemma in inflections:
                    if bundle in inflections[lemma]:
                        inflections[lemma][bundle].append(inflection)
                    else:
                        inflections[lemma][bundle] = [inflection]
                else:
                    inflections[lemma] = {bundle: [inflection]}
    return inflections

def load_hypotheses(iso_code, k=None, method="ensemble", pos_sets=["nouns"],
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
    elif method == "unimorph":
        return load_unimorph_inflections(iso_code, pos_sets)
    elif method == "lemmas":
        return load_lemmas_only(iso_code, pos_sets)
    else:
        raise ValueError(f"Invalid method {method}")


    hyps = {}

    for pos_set in pos_sets:
        hyps_path = hyps_dir / f"{iso_code}.{pos_set}.{suffix}.200"
        logging.info(f"Loading inflection hypotheses from {hyps_path}...")
        with open(hyps_path) as f:
            for line in f:
                fields = line.split("\t")
                if len(fields) != 5:
                    # Then it's probably an empty line between bundles
                    continue
                if iso_code in ["zul", "swh"]:
                    # NOTE the ordering of the lemma and the bundle in the
                    # files in /export/a14/yarowsky-lab/gnicolai/G2PHypotheses
                    # is inverted for these languages specifically.
                    bundle, lemma = fields[0].split("+")
                else:
                    try:
                        lemma, bundle = fields[0].split("+")
                    except ValueError as e:
                        logging.info(f"fields: {fields}")
                        if len(fields[0]) == 4:
                            lemma = fields[0]
                        continue
                        raise e

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

    logging.info(f"Loaded {len(hyps)} lexemes from {hyps_path}...")

    return hyps
