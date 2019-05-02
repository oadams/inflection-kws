""" Explore the data. """
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, Union

babel2iso = {
             "103":"ben",
             "104":"pus",
             "105":"tur",
             "202":"swc",
             "205":"kmr",
             "206":"zul",
             "302":"kaz",
             "303":"tel",
             "304":"lit",
             "404":"kat",
            }
babel2name = {
             "103":"bengali",
             "104":"pashto",
             "105":"turkish",
             "202":"swahili",
             "205":"kurmanji",
             "206":"zulu",
             "302":"kazakh",
             "303":"telugu",
             "304":"lithuanian",
             "404":"georgian",
             }

def load_unimorph_inflections(iso_code):
    """ Given an ISO 639-3 language code, returns a mapping from lemmas of that
    language to list of tuples of <inflection, unimorph bundle>.
    """

    # TODO Need to assert that each lemma only occurs in one contiguous
    # grouping.
    print(f"Loading inflections for {iso_code}")

    inflections = defaultdict(set)
    lang_path = Path("raw/unimorph/{}/{}".format(iso_code, iso_code))
    lemma = None
    #seen_lemmas = set()
    with lang_path.open() as f:
        for line in f:
            sp = line.split("\t")
            if len(sp) == 3:
                lemma, inflection, bundle = sp
                #if lemma in seen_lemmas:
                #    print(f"Seen lemma: {lemma}")
                #    assert False
                inflections[lemma].add((inflection, bundle))
            #else:
                # Then it's a blank line or something else and we can say we've
                # "seen" the lemma. Now any subsequent time we see it would be
                # a lemma-ambiguity wich might break subsequent code.
            #    seen_lemmas.add(lemma)
    return inflections

def load_rttm_toks(babel_code, remove_lang_suffix=True):
    """Loads a list of tokens seen in the RTTM file for a given language."""

    toks = []
    rttm_path = Path(f"uam/asr1/exp/tri5_ali_{babel_code}_test_dev10h/rttm")
    with open(rttm_path) as f:
        for line in f:
            fields = line.split() # RTTMs are space-delimited
            ortho = fields[5] # The written form is in field 6.
            if remove_lang_suffix:
                if ortho.endswith(babel_code):
                    ortho = ortho[:-(len(babel_code)+1)].lower()
            toks.append(ortho)
    return toks

def load_dev_toks(babel_code):
    babel_dev_dir = Path(f"/export/babel/data/{babel_code}-{babel2name[babel_code]}/release-current/conversational/dev/transcription")
    if babel_code == "202":
        babel_dev_dir = Path(f"/export/babel/data/{babel_code}-{babel2name[babel_code]}/IARPA-babel{babel_code}b-v1.0d-build/BABEL_OP2_{babel_code}/conversational/dev/transcription")
    elif babel_code in ["205", "302", "303"]:
        babel_dev_dir = Path(f"/export/babel/data/{babel_code}-{babel2name[babel_code]}/IARPA-babel{babel_code}b-v1.0a-build/BABEL_OP2_{babel_code}/conversational/reference_materials/")
    elif babel_code in ["304"]:
        babel_dev_dir = Path(f"/export/babel/data/{babel_code}-{babel2name[babel_code]}/IARPA-babel{babel_code}b-v1.0b-build/BABEL_OP2_{babel_code}/conversational/reference_materials/")
    elif babel_code in ["404"]:
        babel_dev_dir = Path(f"/export/corpora/LDC/LDC2016S12/IARPA_BABEL_OP3_{babel_code}/conversational/reference_materials/")

    babel_dev_toks = []
    for transcript_path in babel_dev_dir.glob("*.txt"):
        with transcript_path.open() as f:
            for line in f:
                babel_dev_toks.extend(
                        [tok.strip().lower() for tok in line.split()
                                             if not (tok.startswith("[")
                                                and tok.endswith("]"))
                                             and not (tok.startswith("<")
                                                and tok.endswith(">"))])
    return babel_dev_toks

def load_lexicon(babel_code):
    """ Loads the Babel lexicon for a given language. """

    lang_path = Path(f"/export/babel/data/{babel_code}-{babel2name[babel_code]}/release-current/conversational/reference_materials/")
    if babel_code == "202":
        lang_path = Path(f"/export/babel/data/{babel_code}-{babel2name[babel_code]}/IARPA-babel{babel_code}b-v1.0d-build/BABEL_OP2_202/conversational/reference_materials/")
    elif babel_code in ["205", "302", "303"]:
        lang_path = Path(f"/export/babel/data/{babel_code}-{babel2name[babel_code]}/IARPA-babel{babel_code}b-v1.0a-build/BABEL_OP2_{babel_code}/conversational/reference_materials/")
    elif babel_code in ["304"]:
        lang_path = Path(f"/export/babel/data/{babel_code}-{babel2name[babel_code]}/IARPA-babel{babel_code}b-v1.0b-build/BABEL_OP2_{babel_code}/conversational/reference_materials/")
    elif babel_code in ["404"]:
        lang_path = Path(f"/export/corpora/LDC/LDC2016S12/IARPA_BABEL_OP3_{babel_code}/conversational/reference_materials/")
    lexicon_subtrain_path = lang_path / "lexicon.sub-train.txt"
    lexicon_path = lang_path / "lexicon.txt"

    types = set()
    with open(lexicon_path) as f:
        for line in f:
            types.add(line.split("\t")[0])
    #subtrain_types = set()
    #with open(lexicon_subtrain_path) as f:
    #    for line in f:
    #        subtrain_types.add(line.split("\t")[0])

    # Let's assess coverage of the train/dev data with these lexicons.
    dev_toks = load_dev_toks(babel_code)
    print(dev_toks[:100])
    print(len(dev_toks))
    dev_types = set(dev_toks)

    print(len(dev_types.intersection(types))/len(dev_types))
    return types

def load_garrett_hypotheses(iso_code):

    hyps_path = Path(f"/export/a14/yarowsky-lab/gnicolai/G2PHypotheses/{iso_code}.nouns.out")
    if iso_code == "swc":
        # Then use *swh* *verbs* instead.
        hyps_path = Path(f"/export/a14/yarowsky-lab/gnicolai/G2PHypotheses/swh.verbs.out")
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

    #import pprint
    #pprint.pprint(hyps)

    return hyps

def construct_test_set(babel_code):
    """ Constructs a KW test set

    Some key considerations:
        - Every inflection of a paradigm that is observed in the dev set also
        needs to be covered in the pronunciation lexicon. If this isn't
        satisfied, then the RTTM reference might be missing some forms not
        covered by the lexicon, which would mean the "ground truth" is wrong.
    """

    # Find all the dev tokens.
    dev_toks = load_dev_toks(babel_code)
    dev_types = set(dev_toks)

    lexicon = load_lexicon(babel_code)

    covered_lexeme_count = 0
    unimorph_lexemes = load_unimorph_inflections(babel2iso[babel_code])
    covered_lexemes = dict()
    for lemma in unimorph_lexemes:

        # Flag to say whether the lexeme is covered by the pronunciation
        # lexicon
        lexeme_covered = True

        seen_inflections = []
        for inflection, bundle in unimorph_lexemes[lemma]:
            if inflection in dev_types:
                if inflection not in lexicon:
                    # Then we can't use the lexeme, since the RTTM will be missing
                    # the a valid inflection that a system might look for and find.
                    # TODO perhaps confirm that the RTTM doesn't have it.
                    lexeme_covered = False
                seen_inflections.append(inflection)

        if lexeme_covered:
            covered_lexeme_count += 1
            covered_lexemes[lemma] = seen_inflections

    print(f"Covered lexemes: {covered_lexeme_count}")
    print(f"Total lexemes: {len(unimorph_lexemes.keys())}")

    total_seen_inflections = 0
    total_inflections = 0
    for lemma in covered_lexemes:
        total_seen_inflections += len(covered_lexemes[lemma])
        total_inflections += len(unimorph_lexemes[lemma])

    print("Avg. # seen inflections per lexeme: {}".format(total_seen_inflections/len(list(covered_lexemes.keys()))))
    print("Avg. # unimorph inflections per lexeme: {}".format(total_inflections/len(list(covered_lexemes.keys()))))

    # Now additionally constrain based on Garrett's set.
    hyps = load_garrett_hypotheses(babel2iso[babel_code])
    print(len(set(hyps.keys()).intersection(set(covered_lexemes.keys()))))

    # TODO Not sure where this comes from, but it needs to generalize.
    ecf_fn = f"To be replaced w/ the {babel_code} *.ecf.xml"
    version_str = "Inflection KWS test set 0.1."
    #write_kwlist_xml(babel_code, covered_lexemes, ecf_fn, version_str)



    # Create an inverted index to assess how often an inflection occurs in
    # multiple lexemes.
    # TODO discard lexemes that include ambiguous inflections from our test
    # set.
    ambiguous_inflections = []
    ambiguous_forms = set()
    inflection2lemma = defaultdict(set)
    for lemma in covered_lexemes:
        for inflection in covered_lexemes[lemma]:
            inflection2lemma[inflection].add(lemma)
            if len(inflection2lemma[inflection]) > 1:
                ambiguous_inflections.append(inflection)
                ambiguous_forms.add(inflection)
                print(f"ambiguous inflection->lemmas: {inflection}->{inflection2lemma[inflection]}")
    print(f"len(ambiguous_inflections) = {len(ambiguous_inflections)}")
    #print(f"len(ambiguous_forms) = {len(ambiguous_forms)}")
    print(f"total_seen_inflections: {total_seen_inflections}")
    print(f"%age: {100*len(ambiguous_inflections)/total_seen_inflections}")

    filtered_lexemes = dict()
    for lemma in covered_lexemes:
        contains_ambiguous_inflection = False
        for inflection in covered_lexemes[lemma]:
            if inflection in ambiguous_forms:
                contains_ambiguous_inflection = True
                break
        if not contains_ambiguous_inflection:
            filtered_lexemes[lemma] = set(covered_lexemes[lemma])

    print(f"Had {len(covered_lexemes)} lexemes; now has {len(filtered_lexemes)}")
    with open(f"{babel_code}.kwlist.xml", "w") as f:
        print(kwlist_xml(babel_code, filtered_lexemes, ecf_fn, version_str),
              file=f)

def kwlist_xml(babel_code: str,
               paradigms: Dict[str, Iterable[str]],
               ecf_fn: Union[str,Path],
               version_str: str) -> str:
    """ Writes a Keyword list XML file of words to search for.

        paradigms is a dictionary mapping from a lemma to a paradigm
        where each element of the paradigms is a string denoting one of the
        word forms associated with the lexeme.

        Each KW is given an ID of format inspired but distinct from the
        standard Babel KW lists. It takes the form:
            KW<babel_code>-<paradigm-id>-<wordform-id>
        so that one can see which wordforms correspond to the same
        paradigm/lexeme.
    """

    xml_tags = []
    # Opening tag of document
    xml_tags.append(f"<kwlist ecf_filename=\"{ecf_fn}\""
                    f" language=\"{babel2name[babel_code]}\""
                    f" encoding=\"UTF-8\""
                    f" compareNormalize=\"\""
                    f" version=\"{version_str}\">")

    # Write all the inflected forms as keywords.
    for paradigm_id, lemma in enumerate(sorted(list(paradigms.keys()))):
        for inflection_id, inflection in enumerate(sorted(list(paradigms[lemma]))):
            kwid = f"KW-{paradigm_id}-{inflection_id}"
            xml_tag = (f"<kw kwid=\"{kwid}\">\n\t"
                       f"<kwtext>{inflection}</kwtext>\n"
                       "</kw>")
            xml_tags.append(xml_tag)

    xml_tags.append("</kwlist>")
    return "\n".join(xml_tags)

def compare_rttm_unimorph(babel_code):

    rttm_toks = load_rttm_toks(babel_code)
    rttm_types = set(rttm_toks)
    unimorph_lexemes = load_unimorph_inflections(babel2iso[babel_code])
    unimorph_lemmas = set(unimorph_lexemes.keys())

    unimorph_inflections = set()
    for vals in unimorph_lexemes.values():
        inflections = set([inflection for inflection, bundle in vals])
        unimorph_inflections = unimorph_inflections.union(inflections)

    unimorph_noun_inflections = set()
    for vals in unimorph_lexemes.values():
        noun_inflections = set([inflection for inflection, bundle in vals
                                if bundle.startswith("N;")])
        unimorph_noun_inflections = unimorph_noun_inflections.union(noun_inflections)

    print("Lemma coverage:")
    #print(rttm_toks[:10])
    print(f"len rttm_types: {len(rttm_types)}")
    print(f"len unimorph_lemmas: {len(unimorph_lemmas)}")
    print(f"len unimorph_lemmas \cap rttm_types: {len(unimorph_lemmas.intersection(set(rttm_types)))}")
    print(f"len unimorph_inflections: {len(unimorph_inflections)}")
    print(f"len unimorph_inflections \cap rttm_types: {len(unimorph_inflections.intersection(set(rttm_types)))}")

    print(f"len unimorph_noun_inflections: {len(unimorph_noun_inflections)}")
    print(f"len unimorph_noun_inflections \cap rttm_types: {len(unimorph_noun_inflections.intersection(set(rttm_types)))}")

def explore_babel_unimorph(lang):
    """ Explores how well the unimorph paradigms intersect with the Babel dev
    set. """

    # Load the types seen in the Babel Zulu dev set.
    babel_dev_dir = Path(f"/export/babel/data/{lang}-{babel2name[lang]}/release-current/conversational/dev/transcription")
    babel_types = set()
    for transcript_path in babel_dev_dir.glob("*.txt"):
        with transcript_path.open() as f:
            for line in f:
                babel_types = babel_types.union(set([tok.strip().lower() for tok in line.split()
                                                if not (tok.startswith("[")
                                                        and tok.endswith("]"))]))
    #print(types)

    # Load the Zulu unimorph data
    unimorph_inflections = load_unimorph_inflections(babel2iso[lang])

    #print("Number of toks in Babel dev: {}".format(len(babel_toks)))
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
    #explore_babel_unimorph("202")
    #compare_rttm_unimorph("206")
    #load_lexicon("206")
    for babel_code in babel2name:
        if babel2name[babel_code] not in ["pashto"]:
            construct_test_set(babel_code)
            #input()

    #load_cognate_data()
