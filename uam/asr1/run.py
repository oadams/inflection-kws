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
import subprocess
from subprocess import run
import sys

import babel_iso
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
    parser.add_argument("--filt_dtl",
                        action="store_true", default=False,
                        help=("Flag to indicate whether to filter out from the"
                        " lexicon inflections of the test set lemmas that the"
                        " DTL tool missed. This necessarily assumes"
                        " --custom-kwlist is set to True."))
    parser.add_argument("--test_lang", type=str, default="404",
                        help=("The language to test on. Default is Georgian"
                              " (404)."))
    # TODO --custom-kwlist will always be True, since the default is True
    # (sensible), but calling with the flag also sets to True. Need to instead
    # add an option to explicitly select the default Babel kwlist (which isn't
    # that relevant to our experiments).
    parser.add_argument("--custom_kwlist", default=True,
                        action="store_true")
    args = parser.parse_args()
    if not args.custom_kwlist and args.filt_dtl:
        logging.error("--filt_dtl requires the use of an evaluation set based"
        " on lexemes and their inflections to decide on what to filter out of"
        " the oracle lexicon. It filters out inflections the morphological"
        " inflection tool failed to produce that are in lexemes covered by the"
        " test set.")
        assert False
    return args

def prepare_langs(train_langs, recog_langs, filt_dtl=False):
    """ Prepares training data; creates dict and lang dirs with L.fst
        (pronunciation lexicon) and G.fst (language model).

        train_langs is the set of languages that the acoustic model is to be
        trained on. recog_langs is the set of languages that we want to
        recognize. Language models get trained on recog_langs.

        If filt_dtl is True, then we remove from the lexicon hypotheses that
        were not generated by the DTL model (Garrett Nicolai's model for
        generating inflection hypotheses). This flag should actually
        be set to None if we're not filtering, and instead of True, it should
        actually just take a hypothesized inflection dictionary. That way it
        would be agnostic to the specific model.
   """

    # TODO Break this function into smaller pieces.

    # NOTE It's possible for preparation of recog data to fail, in which case
    # the setup_languages.sh script will fail silently. This has been the case for
    # Zulu.

    """
    # Prepare the data/{lang} directories for each lang. Also trains the
    # language models for the recog_langs in data/{recog_lang}_test
    args = ["./local/setup_languages.sh",
            "--langs", " ".join(train_langs),
            "--recog", " ".join(recog_langs)]
    run(args, check=True)
    """

    # TODO Instead of having filt_dtl as a flag, it should probably
    # take the inflection dictionray so that this function can generalize
    # beyond just DTL. This block is also the majority of the function, so
    # should be broken off into another function.
    if filt_dtl:
        # Then we need to run utils/prepare_lang.sh again, after removing words
        # from lexicon.txt
        for babel_code in recog_langs:
            # TODO Generalize this beyond nouns

            iso_code = babel_iso.babel2iso[babel_code]

            # Read in a list of inflections.
            # TODO generalize this beyond DTL
            dtl_paradigms = inflections.load_dtl_hypotheses(iso_code)

            dict_uni = Path(f"data/{babel_code}_test/data/dict_universal")
            # TODO Generalize this filename so that other inflection hypotheses
            # can be used
            dict_uni_filt = Path(f"data/{babel_code}_test/data/dict_universal_filt_dtl")
            # Now we Create a new dict_universal/ directory that filters only
            # for inflections that were hypothesized, or were not in the
            # evaluation set lexemes.

            dtl_inflections = []
            for lemma in dtl_paradigms:
                for bundle in dtl_paradigms[lemma]:
                    dtl_inflections.extend(dtl_paradigms[lemma][bundle])
            dtl_inflections = set(dtl_inflections)

            # First copy the data
            args = ["rsync", "-av",
                    str(dict_uni)+"/", str(dict_uni_filt)]
            run(args, check=True)
            args = ["rm", "-r", str(dict_uni_filt / "tmp.lang")]
            run(args, check=True)

            # We actually need to output words that were in
            # the lexicon but not the eval set. This means we have a
            # dependency on the eval set here for oracle stuff. TODO This
            # definitely means this block should be broken into another
            # function.
            eval_inflections = kws_eval.keyword_inflections(babel_code)
            eval_inflection_list = []
            for lemma in eval_inflections:
                eval_inflection_list.extend(eval_inflections[lemma])
            eval_inflection_set = set(eval_inflection_list)

            # Then change lexionp.txt, lexicon.txt, and nonsilence_lexicon.txt
            # by filtering out words appropriately.
            for fn in ["lexiconp.txt", "lexicon.txt", "nonsilence_lexicon.txt"]:
                logging.info(f"Filtering {fn}...")
                with open(dict_uni / fn) as dict_f, open(dict_uni_filt / fn, "w") as dict_filt_f:
                    for line in dict_f:
                        ortho, *_ = line.split("\t")
                        if ortho.startswith("<") and ortho.endswith(">"):
                            print(line, file=dict_filt_f, end="")
                        elif ortho in dtl_inflections:
                            print(line, file=dict_filt_f, end="")
                        elif ortho not in eval_inflection_set:
                            print(line, file=dict_filt_f, end="")

            lang_uni_filt = Path(
                    f"data/{babel_code}_test/data/lang_universal_filt_dtl")
            # Create a lang directory based on that filtered dictionary.
            args = ["./utils/prepare_lang.sh",
                    "--share-silence-phones", "true",
                    "--phone-symbol-table", "data/lang_universal/phones.txt",
                    str(dict_uni_filt), "<unk>",
                    str(dict_uni_filt / "tmp.lang"),
                    str(lang_uni_filt)]
            run(args, check=True)

            logging.info("Training LM...")
            data_dir = lang_uni_filt.parent
            # Note that we need to retrain the language using the filtered
            # words.txt file.
            args = ["./local/train_lms_srilm.sh",
                    "--oov-symbol", "<unk>",
                    "--train-text", str(data_dir / "train" / "text"),
                    "--words-file", str(lang_uni_filt / "words.txt"),
                    data_dir, str(data_dir / "srilm_filt_dtl")]
            run(args, check=True)

            logging.info("Converting ARPA LM to G.fst...")
            # Convert the ARPA LM file to an FST.
            args = ["./local/arpa2G.sh",
                    str(data_dir / "srilm_filt_dtl" / "lm.gz"),
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

def mkgraph(lang, filt_dtl=True):
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
    """

    # TODO log more details
    logging.info("mkgraph(): Making the HCLG.fst...")

    # NOTE A lot of these functions assume the lang dir is called {lang}_test.
    test_set = f"{lang}_test"

    lang_dir = f"data/{test_set}/data/lang_universal"
    model_dir = f"exp/chain_cleaned/tdnn_sp"
    graph_dir = f"exp/chain_cleaned/tdnn_sp/{test_set}_graph_lang"
    # TODO This flag should generalize beyond just the DTL case.
    if filt_dtl:
        lang_dir = f"data/{test_set}/data/lang_universal_filt_dtl"
        graph_dir = f"exp/chain_cleaned/tdnn_sp/{test_set}_graph_lang_filt_dtl"
    args = ["./utils/mkgraph.sh",
            # For chain models, we need to set the self-loop scale to 1.0. See
            # http://kaldi-asr.org/doc/chain.html#chain_decoding for details.
            "--self-loop-scale", "1.0",
            lang_dir, model_dir, graph_dir]
    run(args, check=True)

def decode(lang, args, env, filt_dtl=True):
    """ Decode the test set.

        This actually means creating word lattices, not utterance-level
        transcripts. These lattices can then be use to get 1 or n-best
        transcriptions, or be used directly in keyword search where the KWS
        output can weight it's confidence of having found a word based on the
        probabilities in the lattice.
    """

    # TODO log more details
    logging.info("decode(): Making the test utterances...")

    test_set = f"{lang}_test"

    # The directory where the HCLG.fst is stored, where the L and G components
    # (pronunciation lexicon and language model, respectively) appropriately
    # cover the test set.
    graph_dir = f"exp/chain_cleaned/tdnn_sp/{test_set}_graph_lang"
    # TODO Generalize this flag beyond DTL.
    if filt_dtl:
        graph_dir = f"exp/chain_cleaned/tdnn_sp/{test_set}_graph_lang_filt_dtl"

    # The directory that contains the speech features (e.g. MFCCs)
    data_dir = f"data/{test_set}/data/dev10h.pem_hires"

    # The directory that word lattices get written to.
    # NOTE The prefix exp/chain_cleaned/tdnn_sp/ needs a single point of
    # control if we want this script to be able to use other models.
    # TODO Generalize this flag beyond DTL.
    decode_dir = f"exp/chain_cleaned/tdnn_sp/{test_set}_decode"
    if filt_dtl:
        decode_dir = f"exp/chain_cleaned/tdnn_sp/{test_set}_decode_filt_dtl"

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

def wer_score(lang, env, filt_dtl=False):
    """ Scores the WER of the lattices.

        This isn't directly used in keyword search, but it can help in
        debugging because you can tell whether the pipeline up until and
        including decoding is working or not by the WER.
    """

    # NOTE Needs a single point of control, as a number of functions assume
    # this.
    test_set = f"{lang}_test"

    data_dir = f"data/{test_set}/data/dev10h.pem"
    lang_dir = f"data/{test_set}/data/lang_universal"
    # NOTE Again, here's another directory that is identical to previously. We
    # need a single point of control.
    decode_dir =  f"exp/chain_cleaned/tdnn_sp/{test_set}_decode"
    if filt_dtl:
        lang_dir = f"data/{test_set}/data/lang_universal_filt_dtl"
        decode_dir =  f"exp/chain_cleaned/tdnn_sp/{test_set}_decode_filt_dtl"
    cmd = "utils/queue.pl --mem 10G"
    args = ["steps/score_kaldi.sh",
            "--cmd", cmd,
            data_dir, lang_dir, decode_dir]
    run(args, check=True)

def prepare_kws(lang, custom_kwlist=True, filt_dtl=False):
    """ Establish KWS lists and ground truth.

        This probably should only have to change when the KW list and ground
        truth change, not when L.fst or G.fst changes.
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

    lang_dir = f"data/{test_set}/data/lang_universal"
    if filt_dtl:
        lang_dir = f"data/{test_set}/data/lang_universal_filt_dtl"
    data_dir = f"data/{test_set}/data/dev10h.pem"

    # Using our own keyword lists.
    if custom_kwlist:
        # TODO Don't hardcode the paths. Or at least hardcode good paths.
        # I should instead be calling kws_eval.test_set() or whatever the
        # function is called.
        kwlist_file = f"kwlists/{lang}.kwlist.xml"
        out_dir = f"{data_dir}/kwset_custom"
        if filt_dtl:
            out_dir = f"{data_dir}/kwset_custom_filt_dtl"
    else:
        # NOTE Assume the KW list files are in the same directory as the
        # RTTM files. For now, just use KWlist 3.
        # TODO We'll want to make this more than just KWS list 3.. this will
        # require data/404_test/data/dev10h.pem/kws to become kws_kwlist{1,2,3,4},
        # like in the babel/s5d script.
        for kwlist_path in Path(rttm_file).parent.glob("*kwlist3.xml"):
            kwlist_file = str(kwlist_path)
        out_dir = f"{data_dir}/kws"

    # TODO this will also have to generalize to multiple kws sets too.
    args = ["local/search/setup.sh",
            ecf_file,
            rttm_file,
            kwlist_file, data_dir, lang_dir, out_dir]
    run(args, check=True)

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
    run(f"fsts-union scp:{out_dir}/tmp.2/keywords.sorted.scp ark,t:\"|gzip -c >{out_dir}/keywords.fsts.gz\"",
        shell=True, check=True)

def kws(lang, env, re_index=True, custom_kwlist=True, filt_dtl=False):
    """ Run keyword search.

        See kaldi/egs/babel/s5d/local/search/run_search.sh for more details on
        how this works.
    """

    test_set = f"{lang}_test"

    lang_dir = f"data/{test_set}/data/lang_universal"
    if filt_dtl:
        lang_dir = f"data/{test_set}/data/lang_universal_filt_dtl"
    data_dir = f"data/{test_set}/data/dev10h.pem"
    decode_dir =  f"exp/chain_cleaned/tdnn_sp/{test_set}_decode"
    if filt_dtl:
        decode_dir =  f"exp/chain_cleaned/tdnn_sp/{test_set}_decode_filt_dtl"
    # NOTE had issues passing in the env["decode_cmd"] because queue.pl wasn't
    # in the PATH or something, so using hardcoded utils/queue.pl for now.
    cmd = "utils/queue.pl --mem 10G"

    kw_dir = Path(f"{decode_dir}/kws")
    extraid=""
    if custom_kwlist:
        # Then local/search/search.sh will create a "kws_custom" dir.
        extraid = "custom"
        kw_dir = Path(f"{decode_dir}/kwset_custom")
        if filt_dtl:
            kw_dir = Path(f"{decode_dir}/kwset_custom_filt_dtl")
            extraid = "custom_filt_dtl"

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
            #"--min-lmwt",
            #"--max-lmwt",
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
    recog_langs = ["404"]

    # The core steps in the pipeline.
    #prepare_langs(train_langs, recog_langs, filt_dtl=args.filt_dtl)

    #prepare_align()
    #train()

    # TODO Perhaps break this second decoding part off into a separate stage
    # which gets determined by a command line argument. For example, ru.py
    # --stage train would run prepare_langs(), prepare_align() and train(),
    # while --stage decode would do mkgraph(), prepare_test_feats(),
    # prepare_test_ivectors(), and decode(). A third --stage kws would create
    # an index given a specific keyword list and score.

    ##### Preparing decoding #####
    # Make HCLG.fst.
    #mkgraph(args.test_lang, filt_dtl=args.filt_dtl)
    # Prepare MFCCs and CMVN stats.
    #prepare_test_feats(args.test_lang, args, env)
    # Prepare ivectors
    #prepare_test_ivectors(args.test_lang, args, env)

    #decode(args.test_lang, args, env, filt_dtl=args.filt_dtl)

    ##### KWS #####
    #prepare_kws(args.test_lang,
    #            filt_dtl=args.filt_dtl,
    #            custom_kwlist=args.custom_kwlist)
    kws(args.test_lang, env,
        custom_kwlist=args.custom_kwlist,
        filt_dtl=args.filt_dtl)
    #wer_score(args.test_lang, env)

    # NOTE If you want to create another evaluation set that's not based off of
    # the Babel dev10h set, then you need to force align your speech and
    # transcript, and create a Rich Transcription Time Marked (RTTM) file,
    # which says at what time and for what duration each transcribed word
    # appears in the utterance. This is the ground truth against which to score
    # KWS hypotheses. For the EMNLP paper, we didn't end up needing this
    # because our eval sets are all based on the Babel dev10h, for which there
    # are RTTM files on the grid. See conf/lang/*.conf for paths.
    ##### Creating RTTM #####
    import create_eval_set
    #create_eval_set.force_align(args.test_lang, args)
    #create_eval_set.generate_rttm(args.test_lang, args)
