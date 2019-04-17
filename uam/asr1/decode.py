""" Prepares the test data and decodes. """

import argparse
import os
import subprocess
from subprocess import run
import sys

def source(source_fn):
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

def decode():
    pass

if __name__ == "__main__":
    args = get_args()
    env = source("cmd.sh")
    #prepare_test_feats("404_test", args, env)
    prepare_test_ivectors("404_test", "exp/nnet3_cleaned/extractor", args, env)
    decode()
