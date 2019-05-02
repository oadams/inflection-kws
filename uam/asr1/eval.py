""" Creates an evaluation set. """

def construct_test_set(babel_code):
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
    dev_toks = load_dev_toks(babel_code)
    dev_types = set(dev_toks)

    # Load the ground-truth Babel lexicon, our oracle.
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
