""" Interface that allows us to deploy various G2P mechanisms. """

import logging
from pathlib import Path
from subprocess import run

import babel_iso

# Georgian G2P rules from Wikipedia using IPA.
kat_rules = {
                "ა": "ɑ" ,#a # For some reason the Babel pron. lex uses a not A.
                "ბ": "b" ,#b
                "გ": "ɡ" ,#g
                "დ": "d" ,#d
                "ე": "ɛ" ,#E
                "ვ": "v" ,#v
                "ზ": "z" ,#z
                "თ": "tʰ",# th
                "ი": "i" ,#i
                "კ": "kʼ",# k
                "ლ": "l" ,#l
                "მ": "m" ,#m
                "ნ": "n" ,#n
                "ო": "ɔ" ,#O
                "პ": "pʼ",# p>
                "ჟ": "ʒ" ,#Z
                "რ": "r" ,#r
                "ს": "s" ,#s
                "ტ": "tʼ",# t
                "უ": "u" ,#u
                "ფ": "pʰ",# ph
                "ქ": "kʰ",# kh
                "ღ": "ɣ" ,#G
                "ყ": "qʼ",# q
                "შ": "ʃ" ,#S
                "ჩ": "t͡ʃʰ", #tSh
                "ც": "t͡sʰ", #tsh
                "ძ": "d͡z", #dz
                "წ": "t͡sʼ", #ts
                "ჭ": "t͡ʃʼ", #tS>
                "ხ": "x", #x
                "ჯ": "d͡ʒ", #dZ
                "ჰ": "h", #h
            }

# Georgian G2P rules from Wikipedia using IPA.
kat_rules = {
                "ა": "a", # For some reason the Babel pron. lex uses a not A.
                "ბ": "b",
                "გ": "g",
                "დ": "d",
                "ე": "E",
                "ვ": "v",
                "ზ": "z",
                "თ": "th",
                "ი": "i",
                "კ": "k",
                "ლ": "l",
                "მ": "m",
                "ნ": "n",
                "ო": "O",
                "პ": "p>",
                "ჟ": "Z",
                "რ": "r",
                "ს": "s",
                "ტ": "t",
                "უ": "u",
                "ფ": "ph",
                "ქ": "kh",
                "ღ": "G",
                "ყ": "q",
                "შ": "S",
                "ჩ": "tSh",
                "ც": "tsh",
                "ძ": "dz",
                "წ": "ts",
                "ჭ": "tS>",
                "ხ": "x",
                "ჯ": "dZ",
                "ჰ": "h",
                "-": "",
                "ჲ": "i",
                "ჳ": "v i",
                "ჱ": "E i",
            }

#tur_rules = {
#        "A": "a:_",
#        "a": "a:_",
#        "B": "b_",
#        "b": "b_",
#        "C": "dZ_",
#}

# TODO Phonetisaurus G2P and rule based G2P functions should have a consistent
# interface. Currently one takes a babel code while the other takes an ISO
# code.
def phonetisaurus_g2p(babel_code, in_fn, out_fn, train_size=5000):
    """ Performs G2P using phonetisaurus models trained my Matthew Wiesner on
    varying amounts of training data."""

    model_path = Path(f"/export/a15/MStuDy/Matthew/LORELEI/kaldi/egs/"
                      f"universal_acoustic_model/g2p_selection_v2/results/"
                      f"{babel_code}_{babel_iso.babel2name[babel_code]}/"
                      f"Random/budget_{train_size}/trial.1/g2p/g2p.fst")

    args = ["phonetisaurus-apply",
            "--model", str(model_path),
            "--word_list", str(in_fn)]
    with open(out_fn, "w") as out_f:
        run(args, stdout=out_f)

def rule_based_g2p(iso_code, ortho):
    #logging.info(f"Converting {ortho} in language {iso_code} to phones...")
    if iso_code == "kat":
        return " ".join([kat_rules[c] for c in ortho])

    raise NotImplementedError(
            f"ISO 639-3 code '{iso_code}' has no G2P rules implemented.")
