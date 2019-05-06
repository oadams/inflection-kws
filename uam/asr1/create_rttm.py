""" A collection of methods that allow for performing forced alignment and
creating your own Rich Transcription Time Marked (RTTM) file. This would be
useful if the evaluation speech doesn't already have an RTTM file. However, for
the purposes of EMNLP 2019, we ended up making a custom keyword set that uses
the the existing Babel dev10h speech sets, which already have ground-truth RTTM
files. So this isn't necessary unless you want to use different speech.

From what I wrote previously:
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
"""

from pathlib import Path
from subprocess import run

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
    # TODO Almost necessarily don't want to be using dev10h, because RTTMs
    # already exist for that.
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
    """ Generate the RTTM file from forced alignment. """

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
