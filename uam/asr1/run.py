""" Trains a universal acoustic model and uses it for keyword search.

    There are a number of broad stages:
        - Gathering corpora / preparing data
        - HMM-GMM training to get alignments
        - Neural network training with the chain objective.
        - Preparing features for the test data (MFCCs and ivectors)
        - Preparing multilingual bottleneck features
        - Creating a HCLG.fst for decoding.
        - Running decoding to get lattices.
        - Preparing KWS data
        - Performing KWS
        - Optionally perform WER scoring.

    If you want to do it all with sensible default settings, just call this
    script.

    Note it's a work in progress, with hardcoded strings that need to become
    more general.
"""

import argparse
import logging
import os
from pathlib import Path
import string
import subprocess
from subprocess import run
import sys

import babel_iso
import g2p
import inflections
import kws_eval

# TODO Set up logging. This will require some thought on when to simply
# redirect the output of subprocesses into a logfile, versus when to write my
# own log statements. For now we're letting subprocesses just write their
# stdout/stderr to stdout. For all logging from this script directly, we use
# the logging module.

def source(source_fns):
    """ Parses environment variables after sourcing a list of files via Bash.

    Kaldi recipes commonly use files that define lots of environment variables
    in them, such as cmd.sh, path.sh, lang.conf. This script accepts a list of
    the names of such files, sources them via a call to bash, and then parses
    all the environment variables, allowing the python environment to have
    access to the same variables.
    """
    # NOTE Maybe relevant variables should just be stored in a Python file. An
    # argument against that is that some scripts that we will call will expect
    # path.sh, cmd.sh and conf/lang.conf to exist anyway. For now development
    # of this recipe should involve as little unnecessary meddling with the
    # course of Kaldi as possible.

    proc = subprocess.Popen(['bash', '-c', " ".join([f'source {source_fn}; ' for source_fn
                                                     in source_fns] + ["set -o posix; set"])],
                            stdout=subprocess.PIPE)
    keysvals = [line.decode("utf-8").strip().split('=') for line in proc.stdout
             if len(str(line).strip().split('=')) == 2]
    source_env = {key.strip(): val.strip() for key, val in keysvals}
    return source_env

def get_args():
    """ Parse commandline arguments. """

    parser = argparse.ArgumentParser()
    parser.add_argument("-v", "--verbose", dest="logging_level",
                        action="store_const",
                        const=logging.INFO,
                        default=logging.WARNING)
    parser.add_argument("--feat_extract_nj", type=int, default=30,
                        help="Number of jobs to run for feature extraction.")
    parser.add_argument("--decode_nj", type=int, default=12,
                        help="Number of jobs to run for decoding.")
    parser.add_argument("--rm_missing",
                        action="store_true", default=False,
                        help=("Flag to indicate whether to remove from the"
                        " lexicon those inflections (of the test set lemmas) that the"
                        " wordform generation tool missed. This necessarily assumes"
                        " --custom-kwlist is set to True."))
    parser.add_argument("--add_spurious",
                        action="store_true", default=False,
                        help=("Flag to indicate whether to add all the"
                        " inflections from the wordform generator, whether"
                        " correct or not. This necessarily assumes"
                        " --custom-kwlist is set to True."))
    parser.add_argument("--test_lang", type=str, required=True,
                        help=("The language to test on. Eg. '404' (Georgian)."))
    parser.add_argument("--compute_wer", action="store_true",
                        default=False,
                        help=("Compute the WER of decoding against"
                        " transcription. This is not needed for keyword search"
                        " but is useful for debugging, to see if the word"
                        " lattices are good."))
    parser.add_argument("--inflection_method", type=str, default="ensemble",
                        help=("A string indicating the method used for"
                              " inflection hypothesis generation."))
    parser.add_argument("--lm_train_text", type=str, default=None,
                        help=("Specify a text file to use as training data for the"
                        " language model, instead of the default Babel data."))
    parser.add_argument("--k", type=int, default=None,
                        help=("The number of inflections per (lemma, bundle)"
                        " to add to the lexicon."))
    parser.add_argument("--rules_g2p", action="store_true", default=False,
                        help=("Use rules-based G2P instead of phonetisaurus."
                              " See g2p.py"))
    parser.add_argument("--kwset_spurious", action="store_true", default=False,
                        help=("Add spurious inflected forms to the KW list."))
    # TODO --custom-kwlist will always be True, since the default is True
    # (sensible), but calling with the flag also sets to True. Need to instead
    # add an option to explicitly select the default Babel kwlist (which isn't
    # that relevant to our experiments).
    parser.add_argument("--custom_kwlist", default=True,
                        action="store_true")
    args = parser.parse_args()
    if not args.custom_kwlist and (args.remove_missing or args.add_spurious):
        logging.error("--rm_missing requires the use of an evaluation set based"
        " on lexemes and their inflections to decide on what to filter out of"
        " the oracle lexicon. It filters out inflections the morphological"
        " inflection tool failed to produce that are in lexemes covered by the"
        " test set.")
        assert False

    # Prepare an experiment affix based on relevant flags.
    exp_affix = ""
    if args.rm_missing or args.add_spurious:
        exp_affix = f"_inflection-method={args.inflection_method}"
    if args.rm_missing:
        exp_affix = f"{exp_affix}_rm-missing"
    if args.add_spurious:
        exp_affix = f"{exp_affix}_add-spurious"
    if args.rules_g2p:
        exp_affix = f"{exp_affix}_rules-g2p"
    if args.lm_train_text:
        exp_affix = f"{exp_affix}_lm-text={Path(args.lm_train_text).stem}"
    if args.k:
        exp_affix = f"{exp_affix}_k={args.k}"
    args.exp_affix = exp_affix

    return args

def prepare_langs(train_langs, recog_langs):
    """ Prepares training data; creates dict and lang dirs with L.fst
        (pronunciation lexicon) and G.fst (language model).

        train_langs is the set of languages that the acoustic model is to be
        trained on. recog_langs is the set of languages that we want to
        recognize. Language models get trained on recog_langs.
   """

    # TODO Break this function into smaller pieces.

    # NOTE It's possible for preparation of recog data to fail, in which case
    # the setup_languages.sh script will fail silently. This has been the case for
    # Zulu.

    # Prepare the data/{lang} directories for each lang. Also trains the
    # language models for the recog_langs in data/{recog_lang}_test
    args = ["./local/setup_languages.sh",
            "--langs", " ".join(train_langs),
            "--recog", " ".join(recog_langs)]
    run(args, check=True)

def prepare_test_lang(babel_code,
                      hyp_paradigms, eval_paradigms,
                      rm_missing=True, add_spurious=True,
                      lm_train_text=None, rules_g2p=False,
                      exp_affix="_rm_missing_add_spurious"):
    """
    Prepares lang dirs for recog_langs again. This needs to be called after a
    call to prepare_langs. Includes options such as rm_missing add_spurious.

    if rm_missing then we remove from the lexicon hypotheses that
    were not generated by the wordform generator. This flag should actually
    be set to None if we're not filtering, and instead of True, it should
    actually just take a hypothesized inflection dictionary. That way it
    would be agnostic to the specific model.

    add_spurious is similar to rm_missing, except it adds to the lexicon
    hypotheses from the morphological induction tool that were missing from the
    lexicon.
    """

    # TODO Generalize this beyond nouns
    iso_code = babel_iso.babel2iso[babel_code]


    hyp_inflections = []
    for lemma in hyp_paradigms:
        for bundle in hyp_paradigms[lemma]:
            hyp_inflections.extend(hyp_paradigms[lemma][bundle])
    hyp_inflections = set(hyp_inflections)

    # First copy the dictionary data
    #args = ["rsync", "-av",
    #        str(dict_uni)+"/", str(dict_uni_filt)]
    #run(args, check=True)
    #args = ["rm", "-r", str(dict_uni_filt / "tmp.lang")]
    #run(args, check=True)

    # Now we Create a new dict_flp* directory that filters only
    # for inflections that were hypothesized, or were not in the
    # evaluation set lexemes.
    dict_flp = Path(f"data/{babel_code}_test/data/dict_flp")
    dict_flp_filt = Path(f"data/{babel_code}_test/data/dict_flp{exp_affix}")
    dict_uni_filt = Path(f"data/{babel_code}_test/data/dict_universal{exp_affix}")

    args = ["rsync", "-av",
            str(dict_flp)+"/", str(dict_flp_filt)]
    run(args, check=True)

    # We actually need to output words that were in
    # the lexicon but not the eval set. This means we have a
    # dependency on the eval set here for oracle stuff. TODO This
    # definitely means this block should be broken into another
    # function.
    eval_inflection_list = []
    for lemma in eval_paradigms:
        eval_inflection_list.extend(eval_paradigms[lemma])
    eval_inflection_set = set(eval_inflection_list)

    # Lex forms will contain all the word forms in the Babel "oracle" lexicon.
    lex_forms = set()
    with open(dict_flp / "lexicon.txt") as dict_f:
        for line in dict_f:
            ortho, *_ = line.split("\t")
            lex_forms.add(ortho)

    # Then change lexionp.txt, lexicon.txt, and nonsilence_lexicon.txt
    # by filtering out words appropriately.
    if rm_missing:
        for fn in ["lexicon.txt", "nonsilence_lexicon.txt"]:
            logging.info(f"Filtering {fn}...")
            with open(dict_flp / fn) as dict_f, open(dict_flp_filt / fn, "w") as dict_filt_f:
                for line in dict_f:
                    ortho, *_ = line.split("\t")
                    if ortho.startswith("<") and ortho.endswith(">"):
                        print(line, file=dict_filt_f, end="")
                    elif ortho in hyp_inflections:
                        print(line, file=dict_filt_f, end="")
                    elif ortho not in eval_inflection_set:
                        print(line, file=dict_filt_f, end="")

    words_to_g2p = []
    if add_spurious:
        # Then change lexionp.txt, lexicon.txt, and nonsilence_lexicon.txt
        # by adding spurious entries from the inflection tool as appropriate.
        dict_fns = ["lexicon.txt", "nonsilence_lexicon.txt"]
        dict_fs = [open(dict_flp_filt / fn, "a") for fn in dict_fns]
        logging.info(f"Adding spurious entries to lexicons...")
        # Now add spurious forms that were generated, if they're
        # not already in the lexicon.
        added = set()
        for lemma in eval_paradigms:
            for bundle in hyp_paradigms[lemma]:
                for ortho in hyp_paradigms[lemma][bundle]:
                    if ortho not in lex_forms and ortho not in added:
                        added.add(ortho)
                        # TODO Generalize beyond rule based G2P.
                        words_to_g2p.append(ortho)

        g2p_pairs = []
        if rules_g2p:
            if babel_code not in ["404"]:
                raise NotImplementedError("Rules-based G2P not implemented for"
                                          f" language {babel_code}.")
            for ortho in words_to_g2p:
                pronunciation = g2p.rule_based_g2p(babel_iso.babel2iso[babel_code], ortho)
                g2p_pairs.append((ortho, pronunciation))
        else:
            # Then use pretrained phonetisaurus models to do G2P.

            # Write words to G2P to a file
            wordform_path = dict_flp_filt / "spurious-wordforms"
            with open(wordform_path, "w") as f:
                for wordform in words_to_g2p:
                    print(wordform, file=f)

            # Call phonetisaurus on them
            g2p_out_path = f"{wordform_path}.g2p"
            g2p.phonetisaurus_g2p(babel_code, wordform_path, g2p_out_path)

            # Load G2P'd words
            with open(g2p_out_path) as f:
                g2p_pairs = [line.strip().split("\t") for line in f]

        for ortho, pronunciation in g2p_pairs:
            for dict_f in dict_fs:
                print(f"{ortho}\t{pronunciation}", file=dict_f)

        for f in dict_fs:
            f.close()

    # Now we prepare the dict directory. This call will also handle mapping
    # diphthongs to individual phones using maps available in local/phone_maps/
    args = ["./local/prepare_universal_dict.sh",
            "--src", str(dict_flp_filt),
            "--dict", str(dict_uni_filt),
            f"{str(babel_code)}_test"]
    run(args, check=True)

    lang_uni_filt = Path(
            f"data/{babel_code}_test/data/lang_universal{exp_affix}")
    # Create a lang directory based on that filtered dictionary.
    args = ["./utils/prepare_lang.sh",
            "--share-silence-phones", "true",
            "--phone-symbol-table", "data/lang_universal/phones.txt",
            str(dict_uni_filt), "<unk>",
            str(dict_uni_filt / "tmp.lang"),
            str(lang_uni_filt)]
    run(args, check=True)

    data_dir = lang_uni_filt.parent

    if lm_train_text:
        # Preprocess the LM training text to an appropriate format.
        # 1. Add in an utterance-ID column
        # 2. Remove punctuation.
        # 3. Ensure there's no blank lines (Common in Bible data in the format
        # Matt Post used).
        fn = Path(lm_train_text).name
        processed_lm_text_path = Path(data_dir / "train" / fn)
        with open(processed_lm_text_path, "w") as out_f, open(lm_train_text) as in_f:
            for line in in_f:
                if line != "\n": # If it's not an empty line
                    print("STUB_UTT_ID ", end="", file=out_f)
                    print(line.translate(str.maketrans('', '', string.punctuation)),
                          file=out_f, end="")

    else:
        # If training text for the language model hasn't been specified, then
        # default to the Babel textual training data.
        lm_train_text = str(data_dir / "train" / "text")

    logging.info(f"Training LM with {lm_train_text}...")
    # Note that we need to retrain the language using the filtered
    # words.txt file.
    args = ["./local/train_lms_srilm.sh",
            "--oov-symbol", "<unk>",
            "--train-text", lm_train_text,
            "--words-file", str(lang_uni_filt / "words.txt"),
            data_dir, str(data_dir / f"srilm{exp_affix}")]
    run(args, check=True)

    logging.info("Converting ARPA LM to G.fst...")
    # Convert the ARPA LM file to an FST.
    args = ["./local/arpa2G.sh",
            str(data_dir / f"srilm{exp_affix}" / "lm.gz"),
            lang_uni_filt, lang_uni_filt]
    run(args, check=True)

def prepare_align():
    """ Prepares training data by aligning audio data to text."""
    # NOTE Untested (I previously made the call manually; but this should
    # work).

    # HMM-GMM training to get alignments between audio and text
    args = ["./local/get_alignments.sh"]
    run(args, check=True)

    # Re-segment training data to select only the audio that matches the
    # transcripts.
    args = ["./local/run_cleanup_segmentation.sh"]
    run(args, check=True)

def train():
    """ Trains a model."""
    # NOTE Untested (I previously made the call manually; but this should
    # work).

    # TODO Talk to Matthew about using 100-lang bottlneck features. I looked in
    # /export/b09/mwiesner/LORELEI/tasks/uam/asr1_99langs/local/prepare_recog.sh
    # and couldn't find anything. These will improve the multilingual acoustic
    # model substantially.

    args = ["./local/chain/run_tdnn.sh"]
    run(args, check=True)

def adapt():
    """ Adapts an existing acoustic model to data in another language. Not
    something we need for EMNLP."""

    # TODO Read Matthew's recipe and implement:
    # /export/b09/mwiesner/LORELEI/tools/kaldi/egs/sinhala/s5/local/adapt_tdnn.sh

def prepare_test_feats(lang, args, env):
    """ Prepares the test features by extracting MFCCs and pitch features."""

    # NOTE Assumes decoding dev10h.pem
    test_set = f"{lang}_test"
    data_dir = f"data/{test_set}/data/dev10h.pem"
    hires_dir = f"{data_dir}_hires"

    # Copy the data to hires directory. Because we will extract hi resolution
    # MFCC features there.
    run(["utils/copy_data_dir.sh",
         data_dir, hires_dir], check=True)

    # Prepare 43-dimensional MFCC/pitch features.
    log_dir = f"exp/make_hires/{test_set}"
    mfcc_dir = f"{hires_dir}/data"
    run(["steps/make_mfcc_pitch_online.sh",
         "--nj", str(args.feat_extract_nj),
         "--mfcc-config", "conf/mfcc_hires.conf",
         "--cmd", env["train_cmd"],
         hires_dir, log_dir, mfcc_dir],
         stdout=sys.stdout, check=True)
    run(["steps/compute_cmvn_stats.sh", hires_dir], check=True)
    run(["utils/fix_data_dir.sh", hires_dir], check=True)

    # Extract features without pitch. Looks like ivector extraction gets done
    # without pitch features, so it'll be useful to have these.
    nopitch_dir = f"{hires_dir}_nopitch"
    run(["utils/data/limit_feature_dim.sh", "0:39",
         hires_dir, nopitch_dir], check=True)
    run(["steps/compute_cmvn_stats.sh", nopitch_dir,
         f"exp/make_hires/{test_set}_nopitch", mfcc_dir], check=True)
    run(["utils/fix_data_dir.sh", nopitch_dir], check=True)

def prepare_test_ivectors(lang, args, env):
    """ Extract ivectors for the test data."""

    # NOTE We assume the ivector extractor is here.
    ivector_extractor_dir = "exp/nnet3_cleaned/extractor"

    test_set = f"{lang}_test"
    # NOTE More assumptions of dev10.pem
    data_dir = f"data/{test_set}/data/dev10h.pem"
    hires_dir = f"{data_dir}_hires"
    nopitch_dir = f"{hires_dir}_nopitch"

    ivector_dir = f"exp/nnet3_cleaned/ivectors_{test_set}_hires"

    tmp_data_dir = f"{ivector_dir}/{test_set}_hires_nopitch_max2"
    run(["utils/data/modify_speaker_info.sh",
         "--utts-per-spk-max", "2",
         nopitch_dir, tmp_data_dir], check=True)
    run(["steps/online/nnet2/extract_ivectors_online.sh",
          "--cmd", env["train_cmd"],
          "--nj", str(args.feat_extract_nj),
          tmp_data_dir, ivector_extractor_dir, ivector_dir], check=True)

def mkgraph(lang, exp_affix=""):
    """ Prepare the HCLG.fst graph that is used in decoding.

        Kaldi uses a weighted finite-state transducer (WFST) architecture. This
        involves composing four finite-state machines together:
            - H: The acoustic model. Consumes acoustic states and outputs
              context-dependent phones.
            - C: Maps context-dependent phones (eg. triphones) to
              context-independent phones.
            - L: The pronunciation lexicon. This maps phones to orthographic
              words, potentially probabilistically.
            - G: The language model. This ascribes probabilities to sequences
              of words. It can also be a non-probabilistic grammar (hence the
              origin of the 'G').

        The call to mkgraph below prepares this graph by using L.fst and G.fst
        that have already been prepared and are found in lang_dir, along with
        the acoustic model found in model_dir. It outputs the resulting
        HCLG.fst to graph_dir.

        In our case of multilingual acoustic modelling followed by decoding a
        specific language, the L and G are specific to the test language, while
        the model_dir is common to all languages.

        exp_affix (experiment affix) is a string that identifies different
        experimental configurations and is used to disambiguate dict, lang, and
        graph dirs etc.
    """

    # TODO log more details
    logging.info("mkgraph(): Making the HCLG.fst...")

    # NOTE A lot of these functions assume the lang dir is called {lang}_test.
    test_set = f"{lang}_test"

    lang_dir = f"data/{test_set}/data/lang_universal{exp_affix}"
    model_dir = f"exp/chain_cleaned/tdnn_sp"
    graph_dir = f"exp/chain_cleaned/tdnn_sp/{test_set}_graph_lang{exp_affix}"
    args = ["./utils/mkgraph.sh",
            # For chain models, we need to set the self-loop scale to 1.0. See
            # http://kaldi-asr.org/doc/chain.html#chain_decoding for details.
            "--self-loop-scale", "1.0",
            lang_dir, model_dir, graph_dir]
    run(args, check=True)

def decode(lang, args, env, exp_affix=""):
    """ Decode the test set.

        This actually means creating word lattices, not utterance-level
        transcripts. These lattices can then be use to get 1 or n-best
        transcriptions, or be used directly in keyword search where the KWS
        output can weight it's confidence of having found a word based on the
        probabilities in the lattice.

        exp_affix (experiment affix) is a string that identifies different
        experimental configurations and is used to disambiguate dict, lang, and
        graph dirs etc.
    """

    # TODO log more details
    logging.info("decode(): Making the test utterances...")

    test_set = f"{lang}_test"

    # The directory where the HCLG.fst is stored, where the L and G components
    # (pronunciation lexicon and language model, respectively) appropriately
    # cover the test set.
    graph_dir = f"exp/chain_cleaned/tdnn_sp/{test_set}_graph_lang{exp_affix}"

    # The directory that contains the speech features (e.g. MFCCs)
    data_dir = f"data/{test_set}/data/dev10h.pem_hires"

    # The directory that word lattices get written to.
    # NOTE The prefix exp/chain_cleaned/tdnn_sp/ needs a single point of
    # control if we want this script to be able to use other models.
    decode_dir = f"exp/chain_cleaned/tdnn_sp/{test_set}_decode{exp_affix}"

    cmd = env["decode_cmd"]
    # TODO I've hardcoded the cmd variable in a few places now.
    cmd = "utils/queue.pl --mem 10G"

    args = ["./steps/nnet3/decode.sh",
        "--cmd", cmd,
        "--nj", str(args.decode_nj),
        # TODO This online ivector dir needs a single point of control.
        "--online-ivector-dir", f"exp/nnet3_cleaned/ivectors_{test_set}_hires",
        # We're using chain models, so we adjust the acoustic
        # weight (acwt) to be close to optimal. We also scale the acoustic
        # probabilities by 10 before dumping the lattice (post-decode-acwt) to make
        # it match LM scales. See
        # http://kaldi-asr.org/doc/chain.html#chain_decoding for details.
        "--acwt", "1.0",
        "--post-decode-acwt", "10.0",
        # We really want to just get the lattices, we can do KWS scoring
        # separately in another function. (steps/nnet3/decode.sh calls
        # local/score.sh by default, which in this case would run KWS)
        "--skip-scoring", "true",
        graph_dir, data_dir, decode_dir]
    run(args, check=True)

def wer_score(lang, env, exp_affix=""):
    """ Scores the WER of the lattices.

        This isn't directly used in keyword search, but it can help in
        debugging because you can tell whether the pipeline up until and
        including decoding is working or not by the WER.
    """

    # NOTE Needs a single point of control, as a number of functions assume
    # this.
    test_set = f"{lang}_test"

    data_dir = f"data/{test_set}/data/dev10h.pem"
    lang_dir = f"data/{test_set}/data/lang_universal{exp_affix}"
    # NOTE Again, here's another directory that is identical to previously. We
    # need a single point of control.
    decode_dir =  f"exp/chain_cleaned/tdnn_sp/{test_set}_decode{exp_affix}"
    cmd = "utils/queue.pl --mem 10G"
    args = ["steps/score_kaldi.sh",
            "--cmd", cmd,
            data_dir, lang_dir, decode_dir]
    run(args, check=True)

# TODO clarify how this differentiates from kws_eval.create_eval_paradigms.
def prepare_kws(lang, custom_kwlist=True, exp_affix="", kwset_affix="",
                kwset_spurious=True, k=None):
    """ Establish KWS lists and ground truth.

        This probably should only have to change when the KW list and ground
        truth change, not when L.fst or G.fst changes. Except it does currently
        set up directories using an exp_affix. Perhaps this should change.

        exp_affix (experiment affix) is a string that identifies different
        experimental configurations and is used to disambiguate dict, lang, and
        graph dirs etc.
    """

    # NOTE For now I'm assuming we're using official Babel KWS gear: the RTTM
    # file and the ECF file. The default behaviour however is to load a custom
    # keyword search list.

    # Flag that decides whether to use the full language pack (FLP) or the
    # limited language pack (LLP)
    flp = False
    # Link the relevant Babel conf for the language.
    babel_egs_path = Path("conf/lang")
    for path in babel_egs_path.glob(f"{lang}*FLP*" if flp else f"{lang}*LLP*"):
        babel_env = source([str(path)])
    # TODO Remove hardcoding and make a common reference to dev10h for all
    # functions in this script
    rttm_file = babel_env["dev10h_rttm_file"]
    ecf_file = babel_env["dev10h_ecf_file"]

    test_set = f"{lang}_test"

    lang_dir = f"data/{test_set}/data/lang_universal{exp_affix}"
    data_dir = f"data/{test_set}/data/dev10h.pem"

    # Using our own keyword lists.
    if custom_kwlist:
        # TODO Don't hardcode the paths. Or at least hardcode good paths.
        # I should instead be calling kws_eval.test_set() or whatever the
        # function is called.
        if kwset_spurious:
            kwlist_file = f"kwlists/k={k}/{lang}.kwlist.xml"
        else:
            kwlist_file = f"kwlists/{lang}.kwlist.xml"
        out_dir = f"{data_dir}/kwset_custom{kwset_affix}"
    else:
        # NOTE Assume the KW list files are in the same directory as the
        # RTTM files. For now, just use KWlist 3.
        # TODO We'll want to make this more than just KWS list 3.. this will
        # require data/404_test/data/dev10h.pem/kws to become kws_kwlist{1,2,3,4},
        # like in the babel/s5d script.
        for kwlist_path in Path(rttm_file).parent.glob("*kwlist3.xml"):
            kwlist_file = str(kwlist_path)
        out_dir = f"{data_dir}/kws"

    logging.info("Calling local/search/setup.sh")
    # TODO this will also have to generalize to multiple kws sets too.
    args = ["local/search/setup.sh",
            ecf_file,
            # NOTE Going to use the previously generated hitlist
            #rttm_file,
            kwlist_file, data_dir, lang_dir, out_dir]
    run(args, check=True)

    logging.info("Calling local/search/compile_keywords.sh")
    # Now to compile keywords
    args = ["local/search/compile_keywords.sh",
            "--filter", "OOV=0&&Characters>2",
            # TODO this will also have to generalize to multiple kws sets too.
            out_dir, lang_dir, f"{out_dir}/tmp.2"]
            #out_dir, lang_dir, out_dir]
    run(args, check=True)

    # Aggregate the keywords from different lists into one FST.
    run(f"sort {out_dir}/tmp.2/keywords.scp > {out_dir}/tmp.2/keywords.sorted.scp",
        shell=True, check=True)
    run(f"fsts-union scp:{out_dir}/tmp.2/keywords.sorted.scp"
        f" ark,t:\"|gzip -c >{out_dir}/keywords.fsts.gz\"",
        shell=True, check=True)

def kws(lang, env, re_index=True, custom_kwlist=True,
        exp_affix="", kwset_affix=""):
    """ Run keyword search.

        See kaldi/egs/babel/s5d/local/search/run_search.sh for more details on
        how this works.
    """

    test_set = f"{lang}_test"

    lang_dir = f"data/{test_set}/data/lang_universal{exp_affix}"
    data_dir = f"data/{test_set}/data/dev10h.pem"
    decode_dir =  f"exp/chain_cleaned/tdnn_sp/{test_set}_decode{exp_affix}"
    # NOTE had issues passing in the env["decode_cmd"] because queue.pl wasn't
    # in the PATH or something, so using hardcoded utils/queue.pl for now.
    cmd = "utils/queue.pl --mem 10G"

    kw_dir = Path(f"{decode_dir}/kws")
    extraid=""
    if custom_kwlist:
        # Then local/search/search.sh will create a "kwset_custom" dir.
        kw_dir = Path(f"{decode_dir}/kwset_custom{kwset_affix}")
        extraid = f"custom{kwset_affix}"

    # NOTE Need to rm .done.index if I need to re-run indexing. Actually TODO, in
    # general for all these functions I should take a kwarg flag that can be
    # used to force all the processing for the given function from scratch.
    done_index = kw_dir / ".done.index"
    if re_index and done_index.exists():
        args = ["rm", str(done_index)]
        run(args, check=True)

    args = ["./local/search/search.sh",
            "--cmd", cmd,
            # I do not know what the max-states argument does.
            #"--max-states",
            # LM Weights: iterates though a few different LM weights, which
            # basically means we can tune based on this hyperparameter
            "--min-lmwt", "12",
            "--max-lmwt", "12",
            "--extraid", extraid, # Flag to indicate custom KW list.
            lang_dir, data_dir, decode_dir]
    run(args, check=True)

if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=args.logging_level)
    # TODO Source path.sh and conf/lang.conf as well.
    env = source(["cmd.sh"])

    # This is a set of most of the Babel languages, except for 4 held-out
    # cases. The choice here is baed on Matthew Wiesner's universal acoustic
    # model preparation. This can be changed but for now I'm leaving it as is
    # for consistency and since the training languages have demonstrably had
    # good phone coverage.
    # TODO Use a mapping from ISO 639-3 codes to Babel lang codes for
    # readability.
    train_langs = ["101", "102", "103", "104", "105", "106",
                   "202", "203", "204", "205", "206", "207",
                   "301", "302", "303", "304", "305", "306",
                   "401", "402", "403"]
    # NOTE We prepare pronunciation lexicons and LMS for the languages below,
    # but don't use acoustic data.
    recog_langs = train_langs + ["107", "201", "307", "404"]
    #prepare_langs(train_langs, recog_langs)
    #prepare_align()
    #train()

    # Prepare MFCCs and CMVN stats for the test language.
    #prepare_test_feats(args.test_lang, args, env)
    # Prepare ivectors for the test language.
    #prepare_test_ivectors(args.test_lang, args, env)

    # Establish the KW eval list.
    eval_paradigms = kws_eval.create_eval_paradigms(args.test_lang,
                                                    args.inflection_method,
                                                    k=args.k,
                                                    kwset_spurious=args.kwset_spurious,
                                                    write_to_fn=True)

    """
    if args.rm_missing or args.add_spurious or args.lm_train_text:
        # Read in the inflections that were hypothesized. We use these to
        # adjust the lexicon that is used for decoding accordingly.
        # TODO generalize this beyond DTL
        hyp_paradigms = inflections.load_hypotheses(babel_iso.babel2iso[args.test_lang],
                                                    k=args.k,
                                                    method=args.inflection_method)

        # Now prepare the lang directory, with the lexicon and LM.
        prepare_test_lang(args.test_lang, hyp_paradigms, eval_paradigms,
                          rm_missing=args.rm_missing,
                          add_spurious=args.add_spurious,
                          lm_train_text=args.lm_train_text,
                          rules_g2p=args.rules_g2p,
                          exp_affix=args.exp_affix)

    # TODO Perhaps break this second decoding part off into a separate stage
    # which gets determined by a command line argument. For example, run.py
    # --stage train would run prepare_langs(), prepare_align() and train(),
    # while --stage decode would do mkgraph(), prepare_test_feats(),
    # prepare_test_ivectors(), and decode(). A third --stage kws would create
    # an index given a specific keyword list and score.

    ##### Preparing decoding #####
    # Make HCLG.fst.
    mkgraph(args.test_lang, exp_affix=args.exp_affix)

    decode(args.test_lang, args, env, exp_affix=args.exp_affix)
    """

    if args.kwset_spurious:
        kwset_affix = f"_k={args.k}"
    ##### KWS #####
    prepare_kws(args.test_lang,
                exp_affix=args.exp_affix,
                kwset_spurious=args.kwset_spurious,
                k=args.k,
                kwset_affix=kwset_affix,
                custom_kwlist=args.custom_kwlist)
    #kws(args.test_lang, env,
    #    exp_affix=args.exp_affix,
    #    kwset_affix=kwset_affix)

    # Computing the word error rate (WER) can be useful for debugging to see if
    # the word lattices generated in decoding are what is causing problems.
    # (The lattice recall metric in KWS scoring is also useful for this).
    if args.compute_wer:
        wer_score(args.test_lang, env)

    # NOTE If you want to create another evaluation set that's not based off of
    # the Babel dev10h set, then you need to force align your speech and
    # transcript, and create a Rich Transcription Time Marked (RTTM) file,
    # which says at what time and for what duration each transcribed word
    # appears in the utterance. This is the ground truth against which to score
    # KWS hypotheses. For the EMNLP paper, we didn't end up needing this
    # because our eval sets are all based on the Babel dev10h, for which there
    # are RTTM files on the grid. See conf/lang/*.conf for paths.
    ##### Creating RTTM #####
    #import create_rttm
    #create_rttm.force_align(args.test_lang, args)
    #create_rttm.generate_rttm(args.test_lang, args)
