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
import os
from pathlib import Path
import subprocess
from subprocess import run
import sys

# TODO Set up logging. This will require some thought on when to simply
# redirect the output of subprocesses into a logfile, versus when to write my
# own log statements.

def source(source_fns):
    """Parses environment variables after sourcing a list of files via Bash.

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
    # course of Kaldi as possible until deeper understanding of the true
    # reality of nature is attained.

    proc = subprocess.Popen(['bash', '-c', " ".join([f'source {source_fn}; ' for source_fn
                                                     in source_fns] + ["set -o posix; set"])],
                            stdout=subprocess.PIPE)
    keysvals = [line.decode("utf-8").strip().split('=') for line in proc.stdout
             if len(str(line).strip().split('=')) == 2]
    source_env = {key.strip(): val.strip() for key, val in keysvals}
    return source_env

def get_args():
    """Parse commandline arguments"""

    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_extract_nj", type=int, default=30,
                        help="Number of jobs to run for feature extraction.")
    parser.add_argument("--decode_nj", type=int, default=12,
                        help="Number of jobs to run for decoding.")
    args = parser.parse_args()
    return args

def prepare_train(train_langs, recog_langs):
    """Prepares training data.

       train_langs is the set of languages that the acoustic model is to be
       trained on. recog_langs is the set of languages that we want to
       recognize. Language models get trained on these languages.
    """

    # Prepare the training data/{lang} directories fore each lang. Also trains
    # the language models for the recog_langs.
    args = ["./local/setup_languages.sh",
            "--langs", " ".join(train_langs),
            "--recog", " ".join(recog_langs)]
    run(args, check=True)

    # HMM-GMM training to get alignments between audio and text.
    args = ["./local/get_alignments.sh"]
    run(args, check=True)

    # Re-segment training data to select only the audio that matches the
    # transcripts.
    args = ["./local/run_cleanup_segmentation.sh"]
    run(args, check=True)

    # TODO Talk to Matthew about using 100-lang bottlneck features. I looked in
    # /export/b09/mwiesner/LORELEI/tasks/uam/asr1_99langs/local/prepare_recog.sh
    # and couldn't find anything.

def train():
    """Trains a model. """

    # TODO call ./local/chain/run_tdnn.sh
    args = ["./local/chain/run_tdnn.sh"]
    run(args, check=True)

def adapt():
    """Adapts an existing acoustic model to data in another language."""

    # TODO Read Matthew's recipe and implement:
    # /export/b09/mwiesner/LORELEI/tools/kaldi/egs/sinhala/s5/local/adapt_tdnn.sh

def prepare_test_feats(lang, args, env):
    """ Prepares the test features by extracting MFCCs and ivectors."""

    test_set = "{lang}_test"
    data_dir = f"data/{test_set}/data/dev10h.pem"
    hires_dir = f"{data_dir}_hires"

    # Copy the data to _hires. Because we will extract hires features there.
    run(["utils/copy_data_dir.sh",
         data_dir, hires_dir], check=True)

    # Make MFCC pitch
    log_dir = f"exp/make_hires/{test_set}"
    mfcc_dir = f"{hires_dir}/data"
    # Prepare 43-dimensional MFCC/pitch features.
    run(["steps/make_mfcc_pitch_online.sh",
         "--nj", str(args.feat_extract_nj),
         "--mfcc-config", "conf/mfcc_hires.conf",
         "--cmd", env["train_cmd"],
         hires_dir, log_dir, mfcc_dir],
         stdout=sys.stdout, check=True)
    run(["steps/compute_cmvn_stats.sh", hires_dir], check=True)
    run(["utils/fix_data_dir.sh", hires_dir], check=True)
    # Extract features without pitch. Looks like ivector extraction gets done
    # without pitch features so it'll be useful to have these.
    nopitch_dir = f"{hires_dir}_nopitch"
    run(["utils/data/limit_feature_dim.sh", "0:39",
         hires_dir, nopitch_dir], check=True)
    run(["steps/compute_cmvn_stats.sh", nopitch_dir,
         f"exp/make_hires/{test_set}_nopitch", mfcc_dir], check=True)
    run(["utils/fix_data_dir.sh", nopitch_dir], check=True)

def prepare_test_ivectors(lang, ivector_extractor_dir, args, env):

    test_set = "{lang}_test"
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

def mkgraph(lang):
    """Prepare the HCLG.fst graph that is used in decoding.

       Kaldi uses a weighted finite-state transducer (WFST) architecture. This
       involves composing four finite-state machines together:
            - H: The acoustic model. Consumes acoustic states and outputs context-dependent phones.
            - C: Maps context-dependent phones (eg. triphones) to context-independent phones
            - L: The pronunciation lexicon. This maps phones to orthographic words,
              potentially probabilistically.
            - G: The language model. This ascribes probabilities to sequences
              of words. It can also be a non-probabilistic grammar (hence the
              origin of the 'G').

        The call to mkgraph below prepares this graph by using L.fst and G.fst
        found in lang_dir and the acoustic model found in model_dir. It outputs
        the resulting HCLG.fst to graph_dir.

        In our case of multilingual acoustic modelling followed by decoding a
        specific language, the L and G are specific to the test language, while
        the model_dir is common to all languages.
    """

    # NOTE A lot of these functions assume the lang dir is called {lang}_test.
    test_set = "{lang}_test"
    lang_dir = f"data/{test_set}/data/lang_universal"
    model_dir = f"exp/chain_cleaned/tdnn_sp"
    graph_dir = f"exp/chain_cleaned/tdnn_sp/{test_set}_graph_lang"
    args = ["./utils/mkgraph.sh",
            # For chain models, we need to set the self-loop scale to 1.0. See
            # http://kaldi-asr.org/doc/chain.html#chain_decoding for details.
            "--self-loop-scale", "1.0",
            lang_dir, model_dir, graph_dir]
    run(args, check=True)

def decode(lang, args, env):
    """Decode the test set.

    This actually means creating lattices, not utterance-level transcripts.
    """

    test_set = "{lang}_test"

    # The directory where the HCLG.fst is stored, where the L and G components
    # (pronunciation lexicon and language model, respectively) appropriately
    # cover the test set.
    # TODO make the ordering of {test_set} with graph_lang consistent with the
    # ordering of {test_set} with decode for the directory names. This caused a
    # time-consuming bug previously!
    graph_dir = f"exp/chain_cleaned/tdnn_sp/{test_set}_graph_lang"

    # The directory that contains the speech features (e.g. MFCCs)
    data_dir = f"data/{test_set}/data/dev10h.pem_hires"

    # The directory that contains
    decode_dir = f"exp/chain_cleaned/tdnn_sp/decode_{test_set}"

    args = ["./steps/nnet3/decode.sh",
        "--cmd", env["decode_cmd"],
        "--nj", str(args.decode_nj),
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

def wer_score(lang, env):
    """ Scores the WER of the lattices. """

    data_dir = f"data/{lang}_test/data/dev10h.pem"
    lang_dir = f"data/{lang}_test/data/lang_universal"
    decode_dir =  f"exp/chain_cleaned/tdnn_sp/decode_{lang}_test"
    cmd = "utils/queue.pl --mem 10G"
    args = ["steps/score_kaldi.sh",
            "--cmd", cmd,
            data_dir, lang_dir, decode_dir]
    run(args, check=True)

def prepare_kws(lang):
    """ Establish KWS lists and ground truth."""

    # TODO Fill this in when I move beyond the standard babel sets.

    # NOTE For now I'm assuming we're using official Babel KWS gear.
    # Flag that decides whether to use the full language pack (FLP) or the
    # limited language pack (LLP)
    flp = False
    # Link the relevant Babel conf for the language.
    babel_egs_path = Path("conf/lang")
    for path in babel_egs_path.glob(f"{lang}*FLP*" if flp else f"{lang}*LLP*"):
        babel_env = source([str(path)])
    print(babel_env["dev10h_rttm_file"])
    print(babel_env["dev10h_ecf_file"])


    # TODO Remove hardcoding and make a common reference to dev10h for all
    # functions in this script
    rttm_file = babel_env["dev10h_rttm_file"]
    ecf_file = babel_env["dev10h_ecf_file"]
    # TODO We'll want to make this more than just KWS list 3.. this will
    # require data/404_test/data/dev10h.pem/kws to become kws_kwlist{1,2,3,4},
    # like in the babel/s5d script.
    kwlist_file = "/export/babel/data/scoring/IndusDB/IARPA-babel404b-v1.0a_conv-dev/IARPA-babel404b-v1.0a_conv-dev.annot.kwlist3.xml"
    lang_dir = f"data/{lang}_test/data/lang_universal"
    data_dir = f"data/{lang}_test/data/dev10h.pem"
    # TODO this will also have to generalize to multiple kws sets too.
    out_dir = f"{data_dir}/kws"
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
    run(f"sort {out_dir}/tmp.2/keywords.scp > {out_dir}/tmp.2/keywords.sorted.scp", shell=True)
    run(f"fsts-union scp:{out_dir}/tmp.2/keywords.sorted.scp ark,t:\"|gzip -c >{out_dir}/keywords.fsts.gz\"", shell=True, check=True)
    


def kws(lang, env):
    """ Run keyword search.

        See kaldi/egs/babel/s5d/local/search/run_search.sh for more details.
    """

    lang_dir = f"data/{lang}_test/data/lang_universal"
    data_dir = f"data/{lang}_test/data/dev10h.pem"
    decode_dir =  f"exp/chain_cleaned/tdnn_sp/decode_{lang}_test"
    cmd = "utils/queue.pl --mem 10G"

    # NOTE Need to rm .done.index if I need to re-run indexing. Actually, in
    # general for all these functions I should take a kwarg flag that asks to
    # redo it all from scratch.
    args = ["./local/search/search.sh",
            "--cmd", cmd,
            # I do not know what these optional arguments do.
            #"--max-states",
            #"--min-lmwt",
            #"--max-lmwt",
            lang_dir, data_dir, decode_dir]
    run(args, check=True)

if __name__ == "__main__":
    args = get_args()
    # TODO Source path.sh and conf/lang.conf as well.
    env = source(["cmd.sh"])

    # TODO Use a mapping from ISO 639-3 codes to Babel lang codes for
    # readability.
    # This is a set of most of the babel languages, except for 4 held-out
    # cases.
    train_langs = ["101", "102", "103", "104", "105", "106",
                   "202", "203", "204", "205", "206", "207",
                   "301", "302", "303", "304", "305", "306",
                   "401", "402", "403"]
    # NOTE We prepare pronunciation lexicons and LMS for the languages below,
    # but don't use acoustic data.
    #recog_langs = train_langs + ["107", "201", "307", "404"]
    recog_langs = ["206"]
    # The cores steps in the pipeline.
    #prepare_train(train_langs, recog_langs)
    #train()

    test_lang = "206"
    ##### Preparing decoding #####
    # Make HCLG.fst.
    mkgraph(test_lang)
    # Prepare MFCCs and CMVN stats.
    prepare_test_feats(test_lang, args, env)
    # Prepare ivectors
    prepare_test_ivectors(test_lang, "exp/nnet3_cleaned/extractor", args, env)

    decode(test_lang, args, env)
    prepare_kws(test_lang)
    kws(test_lang, env)
    wer_score(test_lang, env)
