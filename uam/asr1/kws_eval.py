""" Given Unimorph and Babel data for a language, create a keyword list for
    searching for inflected forms of a word.

    The task is framed as follows: given a word, find occurrences of that
    word in the speech. However, unlike traditional KWS, the model is
    rewarded for finding any inflection of the word, and penalized if those
    are missed.

    There are a number of steps. First is to examine what words occur both
    in the unimorph data as well as in our Babel data. The Unimorph
    paradigms will give us ground truth as to what words are actually
    inflections of those. We then just find which utterances they occur in
    the Babel dev set, making sure we account for homonyms.

    There's also the cognate task variation on this. This has a similar
    formulation, but is different with regards to construction of the
    evaluation set. For that task, we want to search for some source
    language concepts in target language speech. This is essentially
    cross-lingual keyword search.
"""

from collections import defaultdict
import logging
from pathlib import Path
from typing import Dict, Iterable, Union

from babel_iso import babel2name, babel2iso
import inflections

# The dev data is in inconsistently named directories depending on the
# language, so here's a map from the Babel code to the dev set directory.
RESOURCE_DIRS = {}

for babel_code in ["202"]:
    RESOURCE_DIRS[babel_code] = Path(
            f"/export/babel/data/{babel_code}-{babel2name[babel_code]}/"
            f"IARPA-babel{babel_code}b-v1.0d-build/BABEL_OP2_{babel_code}/"
            "conversational/")
for babel_code in ["205", "302", "303"]:
    RESOURCE_DIRS[babel_code] = Path(
            f"/export/babel/data/{babel_code}-{babel2name[babel_code]}/"
            f"IARPA-babel{babel_code}b-v1.0a-build/BABEL_OP2_{babel_code}/"
            "conversational/")
for babel_code in ["304"]:
    RESOURCE_DIRS[babel_code] = Path(
            f"/export/babel/data/{babel_code}-{babel2name[babel_code]}/"
            f"IARPA-babel{babel_code}b-v1.0b-build/BABEL_OP2_{babel_code}/"
            "conversational/")
for babel_code in ["404"]:
    RESOURCE_DIRS[babel_code] = Path(
            f"/export/corpora/LDC/LDC2016S12/IARPA_BABEL_OP3_{babel_code}/"
            f"conversational/")
for babel_code in ["103", "206"]:
    RESOURCE_DIRS[babel_code] = Path(
            f"/export/babel/data/{babel_code}-{babel2name[babel_code]}/"
            f"/release-current/conversational/")
for babel_code in ["105"]:
    RESOURCE_DIRS[babel_code] = Path(
            f"/export/babel/data/{babel_code}-{babel2name[babel_code]}/"
            f"/release-current-b/conversational/")


def load_babel_dev_toks(babel_code, resource_dirs=RESOURCE_DIRS):
    """ Returns a list of tokens seen in the Babel dev10h set.

        This is needed so that we can construct a keyword evaluation set.
    """

    babel_dev_dir = resource_dirs[babel_code] / "dev/transcription/"
    babel_dev_toks = []
    transc_paths = sorted([transc_path for transc_path in babel_dev_dir.glob("*.txt")])
    for transc_path in transc_paths:
        with transc_path.open() as f:
            for line in f:
                babel_dev_toks.extend(
                    [tok.strip().lower() for tok in line.split()
                                         if not (tok.startswith("[")
                                             and tok.endswith("]"))
                                         and not (tok.startswith("<")
                                              and tok.endswith(">"))])
    return babel_dev_toks

def load_lexicon(babel_code, resource_dirs=RESOURCE_DIRS):
    """ Loads the Babel lexicon for a given language. """

    lexicon_path = resource_dirs[babel_code] / "reference_materials/lexicon.txt"

    types = set()
    with open(lexicon_path) as f:
        for line in f:
            types.add(line.split("\t")[0])

    return types

def load_unimorph_inflections(iso_code, unimorph_dir=Path("../../raw/unimorph")):
    """ Given an ISO 639-3 language code, returns a mapping from lemmas of that
        language to list of tuples of <inflection, unimorph bundle>.
    """

    logging.info(f"Loading inflections for {iso_code}")

    inflections = defaultdict(set)
    lang_path = unimorph_dir / f"{iso_code}/{iso_code}"
    lemma = None
    with open(lang_path) as f:
        for line in f:
            sp = line.split("\t")
            if len(sp) == 3:
                lemma, inflection, bundle = sp
                inflections[lemma].add((inflection, bundle))
    return inflections

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

    logging.info(f"Number of inflections in KW list: {len(xml_tags) - 1}")

    xml_tags.append("</kwlist>")
    return "\n".join(xml_tags)

def create_eval_paradigms(babel_code, inflection_method, write_to_fn=False):
    """ Constructs a KW test set.

        The approach taken is to consider Unimorph paradigms and inflections
        that also appear in the Babel dev10h data. That way morphological tools
        can generate inflections and we can search for them in Babel because
        they actually exist. A couple other constraints:
            - Every inflection of a paradigm that is observed in the dev set
              also needs to be covered in the pronunciation lexicon. This is
              because it's good to be able to have an oracle pronunciation
              lexicon comparison.
            - We disregard lexemes that have inflections that are also found in
              other lexemes. The reason for this is that we don't actually have
              ground truth morphological parses of the Babel data, so our
              ground truth for occurrences of the inflections in a given lexeme
              can't be known.
            - We also constrain on the lemmas that the target model actually
              generated inflections for. This way the pipeline can be adapted
              to Garrett's existing models, rather than the other way round
              though. We have to be careful with this though, because we'll
              need to make sure the KWLists are the same between models.
    """

    # Find all words that occur in the Babel dev10h set (ie. forms that we
    # know actually occur in the speech)
    dev_toks = load_babel_dev_toks(babel_code)
    dev_types = set(dev_toks)

    # Load the ground-truth Babel lexicon, our oracle.
    lexicon = load_lexicon(babel_code)

    # We constrain our eval set to lemmas that are "covered" by the
    # pronunciation lexicon. A 'covered' lexeme is one where every inflection
    # in the lexeme that is seen in the Babel speech is also seen in the
    # lexicon. If a lexeme is not "covered", then we can't include it in the
    # evaluation set, since a morphological system may generate the inflection,
    # but it can't be found in the speech because the oracle lexicon doesn't
    # have it. We want the oracle to actually be an oracle.
    # We also keep in covered_lexemes only those that were seen in the speech.
    unimorph_lexemes = load_unimorph_inflections(babel2iso[babel_code])

    # Now load DTL hyps so we can additionally constrain based on Garrett's DTL
    # set. By this I mean the lemmas that were being used to generate
    # inflections, not the inflections themselves.
    dtl_hyps = inflections.load_hypotheses(babel2iso[babel_code],
                                           method=inflection_method)
    logging.info(f"DTL lemmas: {len(set(dtl_hyps.keys()))}")

    covered_lexemes = dict()
    for lemma in unimorph_lexemes:
        lexeme_covered = True
        seen_inflections = []
        for inflection, bundle in unimorph_lexemes[lemma]:
            if inflection in dev_types:
                if inflection not in lexicon:
                    # TODO Confirm that the RTTM doesn't actually have the
                    # form. I'm assuming there's no way it can.
                    logging.info(f"inflection {inflection} of {lemma} not in lexicon")
                    lexeme_covered = False
                    break
                seen_inflections.append(inflection)
        if lexeme_covered and lemma in dtl_hyps:
            # TODO Maybe uncomment the two lines below. We don't really want to
            # report stats on lexemes with no inflections. Note that it
            # shouldn't affect KWS scores, since we just use whatever
            # inflections are available anyway.
            #if seen_inflections == []:
            #   continue
            covered_lexemes[lemma] = seen_inflections
    logging.info(f"# unimorph lexemes: {len(unimorph_lexemes.keys())}")
    logging.info(f"# covered lexemes: {len(covered_lexemes.keys())}")

    total_seen_inflections = 0
    total_inflections = 0
    for lemma in covered_lexemes:
        total_seen_inflections += len(covered_lexemes[lemma])
        total_inflections += len(unimorph_lexemes[lemma])
    logging.info("Avg. # seen inflections per lexeme: {}".format(
            total_seen_inflections/len(list(covered_lexemes.keys()))))
    logging.info("Avg. # unimorph inflections per lexeme: {}".format(
            total_inflections/len(list(covered_lexemes.keys()))))


    # TODO Not sure where this comes from, but it needs to generalize.
    ecf_fn = f"To be replaced w/ the {babel_code} *.ecf.xml"
    version_str = "Inflection KWS test set 0.1."

    # Create an inverted index to assess how often an inflection occurs in
    # multiple lexemes. This is so that we can discard lexemes that include
    # ambiguous inflections from our test set.
    ambiguous_inflections = []
    ambiguous_forms = set()
    inflection2lemma = defaultdict(set)
    for lemma in covered_lexemes:
        for inflection in covered_lexemes[lemma]:
            inflection2lemma[inflection].add(lemma)
            if len(inflection2lemma[inflection]) > 1:
                ambiguous_inflections.append(inflection)
                ambiguous_forms.add(inflection)
                logging.info(f"ambiguous inflection->lemmas: {inflection}->{inflection2lemma[inflection]}")
    logging.info(f"len(ambiguous_inflections) = {len(ambiguous_inflections)}")
    logging.info(f"total_seen_inflections: {total_seen_inflections}")
    logging.info(f"%age: {100*len(ambiguous_inflections)/total_seen_inflections}")

    # Now actually filter out the lexemes with ambiguous inflections.
    filtered_lexemes = dict()
    for lemma in covered_lexemes:
        contains_ambiguous_inflection = False
        for inflection in covered_lexemes[lemma]:
            if inflection in ambiguous_forms:
                contains_ambiguous_inflection = True
                break
        if not contains_ambiguous_inflection:
            filtered_lexemes[lemma] = set(covered_lexemes[lemma])
    logging.info(f"Had {len(covered_lexemes)} lexemes; now has {len(filtered_lexemes)}")

    eval_lexemes = filtered_lexemes

    # Write to a KW list file.
    kwlist_dir = Path("kwlists/")
    if not kwlist_dir.is_dir():
        kwlist_dir.mkdir()
    if write_to_fn:
        with open(kwlist_dir / f"{babel_code}.kwlist.xml", "w") as f:
            print(kwlist_xml(babel_code, eval_lexemes, ecf_fn, version_str),
                  file=f)
    return eval_lexemes
