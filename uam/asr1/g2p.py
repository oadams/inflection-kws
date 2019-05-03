""" Interface that allows us to deploy various G2P mechanisms. """

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
            }

def rule_based_g2p(iso_code, ortho):
    if iso_code == "kat":
        return " ".join([kat_rules[c] for c in ortho])