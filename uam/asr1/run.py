""" Trains a universal acoustic model and uses it for keyword search.

    There are a number of broad stages:
        - Gathering corpora / preparing data
        - HMM-GMM training to get alignments
        - Neural network training with the chain objective.
        - Preparing features for the test data (MFCCs and ivectors)
        - Preparing multilingual bottleneck features
        - Creating a HCLG.fst for decoding.
        - Running decoding to get lattices.

    If you want to do it all with sensible default settings, just call this
    script.
"""

import argparse
import os
import subprocess
from subprocess import run
import sys

# TODO Set up logging. This will require some thought on when to simply
# redirect the output of subprocesses into a logfile, versus when to write my
# own log statements.

def source(source_fn):
    """ Reads from environment variables after sourcing a filename."""
    # NOTE Maybe I should just store relevant variables in a Python file. An
    # argument against that is that some scripts will expect path.sh, cmd.sh
    # and conf/lang.conf to have been sourced.

    proc = subprocess.Popen(['bash', '-c', 'source {} && env'.format(source_fn)],
                            stdout=subprocess.PIPE)
    keysvals = [line.decode("utf-8").strip().split('=') for line in proc.stdout
             if len(str(line).strip().split('=')) == 2]
    source_env = {key.strip(): val.strip() for key, val in keysvals}
    return source_env

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nj", type=int, default=30,
                        help="Number of jobs to run for feature extraction.")
    args = parser.parse_args()
    return args

def prepare_training():
    """ Prepares training data. """

    # TODO call ./local/setup_languages.sh with appropriate environment
    # variables set. See run.sh

    # TODO call ./local/get_alignments.sh with appropriate environment
    # variables set. See run.sh

    # TODO call ./local/run_cleanup_segmentation.sh

    # TODO Talk to Matthew about using 100-lang bottlneck features. I looked in
    # /export/b09/mwiesner/LORELEI/tasks/uam/asr1_99langs/local/prepare_recog.sh
    # and couldn't find anything.

def train():
    """ Trains a model. """

    # TODO call ./local/chain/run_tdnn.sh

def prepare_test_feats(test_set, args, env):
    """ Prepares the test features by extracting MFCCs and ivectors."""

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
         "--nj", str(args.nj),
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

def prepare_test_ivectors(test_set, ivector_extractor_dir, args, env):

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
          "--nj", str(args.nj),
          tmp_data_dir, ivector_extractor_dir, ivector_dir], check=True)

def mkgraph(test_set):
    """ Prepare the HCLG.fst graph that is used in decoding. """

    lang_dir = f"data/{test_set}/data/lang_universal"
    model_dir = f"exp/chain_cleaned/tdnn_sp"
    graph_dir = f"exp/chain_cleaned/tdnn_sp/{test_set}_graph_lang"
    args = ["./utils/mkgraph.sh", lang_dir, model_dir, graph_dir]
    run(args, check=True)

def decode(test_set, env):
    """Decode the test set.

    This actually means creating lattices, not utterance-level transcripts.
    """

    # The directory where the HCLG.fst is stored, where the L and G components
    # (pronunciation lexicon and language model, respectively) appropriately
    # cover the test set.
    graph_dir = f"exp/chain_cleaned/tdnn_sp/{test_set}_graph_lang"

    # The directory that contains the speech features (e.g. MFCCs)
    data_dir = f"data/{test_set}/data/dev10h.pem_hires"

    # The directory that contains
    decode_dir = f"exp/chain_cleaned/tdnn_sp/decode_{test_set}"
    args = ["./steps/nnet3/decode.sh",
        "--cmd", env["decode_cmd"],
        "--online-ivector-dir", f"exp/nnet3_cleaned/ivectors_{test_set}_hires",
        "--post-decode-acwt", "10.0",
        graph_dir, data_dir, decode_dir]
    print(args)
    run(args, check=True)

def prepare_kws():
    """ Establish KWS lists and ground truth."""

    # TODO Fill this in when I move beyond the standard babel sets.
    pass

def kws():
    """ Run keyword search. """

    # TODO Create inverted index

    # Get scores.

def runcheck(*args, **kwargs):
    """
    Wrapper to subprocess.run that calls it with check=True so that failed
    subprocesses raise an exception.
    """
    kwargs["check"] = True
    try:
        subprocess.run(*args, **kwargs)
    except subprocess.CalledProcessError as e:
        print("")

if __name__ == "__main__":
    args = get_args()
    # TODO Source path.sh and conf/lang.conf as well.
    env = source("cmd.sh")

    # The cores steps in the pipeline.
    #prepare_train()
    #train()

    ##### Preparing decoding #####
    # Make HCLG.fst.
    #mkgraph("404_test")
    # Prepare MFCCs and CMVN stats.
    #prepare_test_feats(test_set, args, env)
    # Prepare ivectors
    #prepare_test_ivectors(test_set, "exp/nnet3_cleaned/extractor", args, env)

    decode("404_test", env)
    #prepare_kws()
    #kws()
