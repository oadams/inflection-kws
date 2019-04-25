""" Explore the data. """
from collections import defaultdict
from pathlib import Path

def load_unimorph_inflections(lang):
    """ Given an ISO 639-3 language code, returns a mapping from lemmas of that
    language to list of tuples of <inflection, unimorph bundle>.
    """

    inflections = defaultdict(list)
    lang_path = Path("raw/unimorph/{}/{}".format(lang, lang))
    with lang_path.open() as f:
        for line in f:
            sp = line.split()
            if len(sp) == 3:
                lemma, inflection, bundle = sp
                inflections[lemma].append((inflection, bundle))
    return inflections

def explore_babel_unimorph():
    """ Explores how well the unimorph paradigms intersect with the Babel dev
    set. """

    # Load the types seen in the Babel Zulu dev set.
    babel_dev_dir = Path("/export/babel/data/206-zulu/release-current/conversational/dev/transcription")
    babel_types = set()
    for transcript_path in babel_dev_dir.glob("*.txt"):
        with transcript_path.open() as f:
            for line in f:
                babel_types = babel_types.union(set([tok.strip().lower() for tok in line.split()
                                                if not (tok.startswith("[")
                                                        and tok.endswith("]"))]))
    #print(types)

    # Load the Zulu unimorph data
    unimorph_inflections = load_unimorph_inflections("zul")

    print("Number of types in Babel dev: {}".format(len(babel_types)))
    print("Number of lemmas in Unimorph: {}".format(len(unimorph_inflections)))
    print("card(Babel & Unimorph): {}".format(len(set(unimorph_inflections.keys()).intersection(babel_types))))

def print_pos_statistics(unimorph_inflections):
    """ Print statistics about the unimorph inflections. """

    pos_counts = defaultdict(int)
    for lemma in unimorph_inflections:
        # Assumes the first inflection, at index 0, is representative of all.
        _, bundle = unimorph_inflections[lemma][0]
        bundle = bundle.split(";")
        pos = bundle[0]
        pos_counts[pos] += 1

    print(pos_counts)
    num_inflections = defaultdict(list)
    for lemma in unimorph_inflections:
        _, bundle = unimorph_inflections[lemma][0]
        bundle = bundle.split(";")
        pos = bundle[0]
        num_inflections[pos].append(len(unimorph_inflections[lemma]))
    avg_num_inflections = dict()

    for pos in num_inflections:
        avg_num_inflections[pos] = sum(num_inflections[pos])/len(num_inflections[pos])

    print(avg_num_inflections)

def load_cognate_data():
    """Loads Winston's ground truth cognate data scraped from Wiktionary."""

    from iso639 import languages
    import pandas as pd

    cog_fn = Path("raw/cognates.txt")
    babel_langs_fn = Path("raw/babel_langs.tsv")

    lang2types = defaultdict(set)
    lang2toks = defaultdict(list)
    with open(cog_fn) as f:
        for line in f:
            src, _, tgt = line.split("\t")
            src_lang, src_type = src.split("|")
            tgt_lang, tgt_type = tgt.split("|")
            lang2types[src_lang].add(src_type)
            lang2types[tgt_lang].add(tgt_type)
            lang2toks[src_lang].append(src_type)
            lang2toks[tgt_lang].append(tgt_type)

    babel_df = pd.read_csv(str(babel_langs_fn), sep="\t")

    unimorph_df = babel_df[babel_df["In Unimorph"] == True]

    for code in babel_df["iso 639-1"]:
        if (len(lang2toks[code]) > 0 and
                code in unimorph_df["iso 639-1"].values):
            print(f"code: {code}, #types: {len(lang2types[code])}, #toks: {len(lang2toks[code])}")
    for code in babel_df["iso 639-3"]:
        if (len(lang2toks[code]) > 0 and
                code in unimorph_df["iso 639-3"].values):
            print(f"code: {code}, #types: {len(lang2types[code])}, #toks: {len(lang2toks[code])}")
    """
        try:
            if len(code) == 2:
                lang = languages.get(alpha2=code)
            elif len(code) == 3:
                lang = languages.get(terminology=code)
            print(f"code: {code}, name: {lang.name}")
        except KeyError:
            print(f"Couldn't find code: {code}")
    """


if __name__ == "__main__":
    lang = "zul"
    #unimorph_inflections = load_unimorph_inflections(lang)
    #print_pos_statistics(unimorph_inflections)

    load_cognate_data()
