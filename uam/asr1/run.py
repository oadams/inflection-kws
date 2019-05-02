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

import babel_iso
import inflections

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

def prepare_langs(train_langs, recog_langs, filter_morph_hyps=True):
    """Prepares training data.

       train_langs is the set of languages that the acoustic model is to be
       trained on. recog_langs is the set of languages that we want to
       recognize. Language models get trained on these languages.
    """

    """
    # NOTE It's possible for recog data being prepared to fail, and the
    # setup_languages.sh script will fail silently. This has been the case for
    # Zulu.

    # Prepare the data/{lang} directories for each lang. Also trains the
    # language models for the recog_langs in data/{recog_lang}_test
    args = ["./local/setup_languages.sh",
            "--langs", " ".join(train_langs),
            "--recog", " ".join(recog_langs)]
    run(args, check=True)
    """

    # 
    if filter_morph_hyps:
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
            # Create a new dict_universal that filters only for inflections
            # that were hypothesized

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
            # Then change lexionp.txt, lexicon.txt, and nonsilence_lexicon.txt
            for fn in ["lexiconp.txt", "lexicon.txt", "nonsilence_lexicon.txt"]:
                with open(dict_uni / fn) as dict_f, open(dict_uni_filt / fn, "w") as dict_filt_f:
                    for line in dict_f:
                        ortho, *_ = line.split("\t")
                        if ortho.startswith("<") and ortho.endswith(">"):
                            print(line, file=dict_filt_f, end="")
                            continue
                        if ortho in dtl_inflections:
                            print(line, file=dict_filt_f, end="")
                        # TODO We actually need to print words that were in the
                        # lexicon but not the eval set... so we have a
                        # dependency on the eval set here for oracle stuff.

            """
            lang_uni_filt = Path(f"data/{babel_code}_test/data/lang_universal_filt_dtl")
            # Create a lang directory based on that filtered dictionary.
            args = ["./utils/prepare_lang",
                    "--share-silence-phones", "true",
                    "--phone-symbol-table", "data/lang_universal/phones.txt",
                    str(dict_uni_filt), "<unk>",
                    str(dict_uni_filt / "tmp.lang"),
                    str(lang_uni_filt)]
            run(args, check=True)
            """

            # Need to create a language model too. Can just copy the unfiltered
            # one?

def prepare_align():
    """Prepares training data by aligning audio data to text."""
    # NOTE Untested
# HMM-GMM training to get alignments between audio and text.  args = ["./local/get_alignments.sh"] run(args, check=True)

    # Re-segment training data to select only the audio that matches the
    # transcripts.
    args = ["./local/run_cleanup_segmentation.sh"]
    run(args, check=True)

def train():
    """Trains a model."""
    # NOTE Untested

    # TODO Talk to Matthew about using 100-lang bottlneck features. I looked in
    # /export/b09/mwiesner/LORELEI/tasks/uam/asr1_99langs/local/prepare_recog.sh
    # and couldn't find anything.

    args = ["./local/chain/run_tdnn.sh"]
    run(args, check=True)

def adapt():
    """Adapts an existing acoustic model to data in another language."""

    # TODO Read Matthew's recipe and implement:
    # /export/b09/mwiesner/LORELEI/tools/kaldi/egs/sinhala/s5/local/adapt_tdnn.sh

def prepare_test_feats(lang, args, env):
    """Prepares the test features by extracting MFCCs and pitch features."""

    # NOTE Assumes decoding dev10h.pem
    test_set = f"{lang}_test"
    data_dir = f"data/{test_set}/data/dev10h.pem"
    hires_dir = f"{data_dir}_hires"

    # Copy the data to _hires. Because we will extract hires features there.
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
    # without pitch features so it'll be useful to have these.
    nopitch_dir = f"{hires_dir}_nopitch"
    run(["utils/data/limit_feature_dim.sh", "0:39",
         hires_dir, nopitch_dir], check=True)
    run(["steps/compute_cmvn_stats.sh", nopitch_dir,
         f"exp/make_hires/{test_set}_nopitch", mfcc_dir], check=True)
    run(["utils/fix_data_dir.sh", nopitch_dir], check=True)

def prepare_test_ivectors(lang, args, env):
    """Extract ivectors for the test data."""

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
    test_set = f"{lang}_test"

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

    test_set = f"{lang}_test"

    # The directory where the HCLG.fst is stored, where the L and G components
    # (pronunciation lexicon and language model, respectively) appropriately
    # cover the test set.
    graph_dir = f"exp/chain_cleaned/tdnn_sp/{test_set}_graph_lang"

    # The directory that contains the speech features (e.g. MFCCs)
    data_dir = f"data/{test_set}/data/dev10h.pem_hires"

    # The directory that contains
    # NOTE Needs a single point of control.
    decode_dir = f"exp/chain_cleaned/tdnn_sp/{test_set}_decode"

    cmd = env["decode_cmd"]
    cmd = "utils/queue.pl --mem 10G"

    args = ["./steps/nnet3/decode.sh",
        "--cmd", cmd,
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

    # NOTE Needs a single point of control.
    test_set = f"{lang}_test"

    data_dir = f"data/{test_set}/data/dev10h.pem"
    lang_dir = f"data/{test_set}/data/lang_universal"
    # NOTE Again, here's another directory that is identical to previously. We
    # need a single point of control.
    decode_dir =  f"exp/chain_cleaned/tdnn_sp/{test_set}_decode"
    cmd = "utils/queue.pl --mem 10G"
    args = ["steps/score_kaldi.sh",
            "--cmd", cmd,
            data_dir, lang_dir, decode_dir]
    run(args, check=True)

def prepare_kws(lang, custom_kwlist=True):
    """ Establish KWS lists and ground truth."""

    # NOTE For now I'm assuming we're using official Babel KWS gear.
    # Flag that decides whether to use the full language pack (FLP) or the
    # limited language pack (LLP)
    flp = False
    # Link the relevant Babel conf for the language.
    babel_egs_path = Path("conf/lang")
    for path in babel_egs_path.glob(f"{lang}*FLP*" if flp else f"{lang}*LLP*"):
        babel_env = source([str(path)])
    test_set = f"{lang}_test"


    # TODO Remove hardcoding and make a common reference to dev10h for all
    # functions in this script
    rttm_file = babel_env["dev10h_rttm_file"]
    ecf_file = babel_env["dev10h_ecf_file"]

    lang_dir = f"data/{test_set}/data/lang_universal"
    data_dir = f"data/{test_set}/data/dev10h.pem"

    # Block for using our own keyword lists.
    if custom_kwlist:
        # TODO Don't hardcode the paths. Or at least hardcode good paths.
        kwlist_file = f"../../{lang}.kwlist.xml"
        out_dir = f"{data_dir}/kwset_custom"
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

def kws(lang, env, custom_kwlist=True):
    """Run keyword search.

       See kaldi/egs/babel/s5d/local/search/run_search.sh for more details.
    """

    test_set = f"{lang}_test"

    lang_dir = f"data/{test_set}/data/lang_universal"
    data_dir = f"data/{test_set}/data/dev10h.pem"
    decode_dir =  f"exp/chain_cleaned/tdnn_sp/{test_set}_decode"
    # NOTE had issues passing in the env["decode_cmd"] because queue.pl wasn't
    # in the PATH or something, so using hardcoded utils/queue.pl for now.
    cmd = "utils/queue.pl --mem 10G"

    extraid=""
    if custom_kwlist:
        # Then local/search/search.sh will create a "kws_custom" dir.
        extraid = "custom"

    # NOTE Need to rm .done.index if I need to re-run indexing. Actually TODO, in
    # general for all these functions I should take a kwarg flag that can be
    # used to force all the processing for the given function from scratch.
    args = ["./local/search/search.sh",
            "--cmd", cmd,
            # I do not know what these optional arguments do.
            #"--max-states",
            #"--min-lmwt",
            #"--max-lmwt",
            "--extraid", extraid,
            lang_dir, data_dir, decode_dir]
    run(args, check=True)

def add_lang_suffixes_to_tokens(transcription_path, lang):
    """ Replaces words in a Kaldi-style transcription with a suffix.

    For example, given lang="202", the Babel Swahili utterance
        10524_A_20131009_200043_004061 <hes> kuoa ndiyo nilikuwa
        nimeoa
    would become:
        10524_A_20131009_200043_004061 <hes> kuoa_202 ndiyo_202 nilikuwa_202
        nimeoa_202
    Notice that the <hes> tag was unaffected. Things in angle brackets such as
    <hes> and <noise> are ignored as they are considered language independent.
    """

    with open(transcription_path) as f:
        lines = [line.split() for line in f]

    with open(transcription_path, "w") as f:
        for line in lines:
            utt_id = line[0]
            transcript_toks = []
            for tok in line[1:]:
                if tok.startswith("<") and tok.endswith(">"):
                    transcript_toks.append(tok)
                else:
                    transcript_toks.append(f"{tok}_{lang}")
            line_out = " ".join([utt_id] + transcript_toks)
            print(line_out, file=f)

def force_align(lang, args):
    """

    Note that if the evaluation keywords are in the dev10h set, there's no need
    to do this force_align / generate_rttm() step, because there are gold
    standard RTTM files listed in conf/lang.conf.
    """

    nj = str(args.decode_nj)

    # TODO Retrain the tri5 model using both train and dev data in training so
    # we can get better ground-truth alignments.

    # TODO Possibly we want to rejig train data so that we have no OOV tokens
    # in our dev set, so that we can get better ground truth for our newly
    # generated KWS tasks.

    # First perform forced alignment
    # TODO Make sure this is the right alignment script. Can I just use what
    # we used for HMM-GMM alignment in preprocessing?
    test_set = f"{lang}_test"
    data_dir = f"data/{test_set}/data/dev10h.pem"

    # The multilingual training data suffixed each word with the lang-code, to
    # make language-specific representations of homographs across languages.
    # However, the dev transcription data does not include such word suffixes,
    # so there is a lexicon mismatch. One solution would be to change how trainin
    # graphs are compiled. Another approach is instead to create a new version
    # of the dev data directory where we process the text to have the language
    # suffixes. This is the approach we take here.

    # Make a copy of the data directory so that we can modify the text to be
    # compatible with the training graphs.
    lang_suffix_data_dir = f"{data_dir}.lang_suffix"
    args = ["rsync", "-av", f"{data_dir}/", lang_suffix_data_dir]
    del data_dir # So we don't accidentally reference it.
    run(args, check=True)
    # Delete the data splits in that directory that have the old token form.
    split_nj_path = Path(f"{lang_suffix_data_dir}/split{nj}")
    if split_nj_path.exists():
        args = ["rm", "-r", str(split_nj_path)]
        run(args, check=True)
    # 3. Replace word tokens with word_{lang}
    add_lang_suffixes_to_tokens(Path(lang_suffix_data_dir) / "text", lang)

    # NOTE This was the model used in getting the initial alignments for
    # training.
    lang_dir = "data/lang_universalp/tri5"
    # TODO Exp dir an ali dir will have to change to be relevant to the
    # specific language at hand.
    src_dir = f"exp/tri5" # The directory that has the AM we'll use for alignment.
    ali_dir = f"exp/tri5_ali_{test_set}_dev10h" # TODO Really need to factor out dev10h

    cmd = "utils/queue.pl --mem 10G"
    args = ["steps/align_fmllr.sh",
            "--nj", nj,
            "--cmd", cmd,
            lang_suffix_data_dir, lang_dir, src_dir, ali_dir]
    run(args, check=True)

def generate_rttm(lang, args):
    """
    """

    test_set = f"{lang}_test"

    # Now produce an RTTM file from the alignments
    # TODO Generalize these.
    #data_dir = f"data/{test_set}/data/dev10h.pem"
    #lang_dir = f"data/{test_set}/data/lang_universal"
    #ali_dir = f""
    data_dir = f"data/{test_set}/data/dev10h.pem"
    lang_suffix_data_dir = f"{data_dir}.lang_suffix"
    lang_dir = "data/lang_universalp/tri5"
    ali_dir = f"exp/tri5_ali_{test_set}_dev10h"
    args = ["local/ali_to_rttm.sh",
            lang_suffix_data_dir, lang_dir, ali_dir]
    run(args, check=True)

    # NOTE In order to have a lexicon that dealt with things like t_"_B, I
    # needed to run cut -d " " -f 2- \
    # data/lang_universalp/tri5/phones/align_lexicon.txt > \
    # data/lang_universalp/tri5/phones/lexicon.txt
    # On top of this, the call in ali_to_rttm.sh to local/make_L_align.sh had
    # to change the first argument to data/lang_universalp/tri5/phones

def create_unimorph_babel_kwlist(lang):
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

        Next we need to get a finer-grained location of those words in the
        utterance. To do this we need to run forced alignment between the
        utterance and the speech, probably using the HMM-GMM system that was
        used to perform alignment before training the chain model in the
        first place. Actually, we will want to retrain the HMM-GMM system,
        incorporating the dev10 data into the training data so that we can get
        'ground-truth' alignments. But as a first approximation, reusing the
        existing tri5 model will probably do. Alignments should be accurate,
        since we do in fact get to see the transcription. With those alignments
        in place, we can create the requisite RTTM, ECF and KW list XML files.
        Then we simply call prepare_kws() and kws() as previously.

        There's also the cognate task variation on this. This has a similar
        formulation, but is different with regards to construction of the
        evaluation set. For that task, we want to search for some source
        language concepts in target language speech. This is essentially
        cross-lingual keyword search.
    """

    # First generate a set of words, along with the utterances that their
    # inflections are found in.
    #x = generate_unimorph_babel_words()

    # Then generate the RTTM file, which represents where in each utterance the
    # keyword or its inflections occur. This will involve a call to
    # egs/babel/local/ali_to_rttm.sh
    force_align(lang, args)
    generate_rttm(lang)

    # Generate the ECF file, which represents....TODO
    #generate_ecf(x)

if __name__ == "__main__":
    args = get_args()
    # TODO Source path.sh and conf/lang.conf as well.
    env = source(["cmd.sh"])

    # This is a set of most of the Babel languages, except for 4 held-out
    # cases.
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
    prepare_langs(train_langs, recog_langs)

    #prepare_align()
    #train()

    # TODO Perhaps break this second decoding part off into a separate stage
    # which gets determined by a command line argument. For example, ru.py
    # --stage train would run prepare_langs(), prepare_align() and train(),
    # while --stage decode would do mkgraph(), prepare_test_feats(),
    # prepare_test_ivectors(), and decode(). A third --stage kws would create
    # an index given a specific keyword list and score.

    test_lang = "404"

    """
    ##### Preparing decoding #####
    # Make HCLG.fst.
    mkgraph(test_lang)
    # Prepare MFCCs and CMVN stats.
    prepare_test_feats(test_lang, args, env)
    # Prepare ivectors
    prepare_test_ivectors(test_lang, args, env)

    decode(test_lang, args, env)
    """

    """
    ##### Creating RTTM #####
    force_align(test_lang, args)
    generate_rttm(test_lang, args)
    """

    """
    custom_kwlist=True
    ##### KWS #####
    #prepare_kws(test_lang, custom_kwlist=custom_kwlist)
    kws(test_lang, env, custom_kwlist=custom_kwlist)
    #wer_score(test_lang, env)
    """
