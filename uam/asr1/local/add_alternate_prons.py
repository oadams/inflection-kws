from __future__ import print_function
import argparse
import sys
import os
import codecs

UTF8Reader = codecs.getreader('utf8')
sys.stdin = UTF8Reader(sys.stdin)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("lex", help="path to lexicon")
    parser.add_argument("phone_map", help="path with list of alternate phones", nargs="?", default='')
    args = parser.parse_args()


    phone_map = {}
    if (args.phone_map == ''):
        for l in sys.stdin:
            p, p_alt = l.strip().split(None, 1)
            phone_map[p] = p_alt
    else:
        with codecs.open(args.phone_map, "r", encoding="utf-8") as f:
            for l in f:
                p, p_alt = l.strip().split(None, 1)
                phone_map[p] = p_alt

    changeable_phones = set(phone_map.keys())

    lex = []
    with codecs.open(args.lex, "r", encoding="utf-8") as f:
        for l in f:
            w, pron = l.strip().split(None, 1)
            change_phones = set(pron.split()).intersection(changeable_phones)
            lex.append((w, pron))
            
            # We don't do every combinatorial possiblity as this would get out
            # of hand. Instead we simply make a single alternative pron for
            # each phoneme that could be pronounced differently
            #
            # E.g. Cart c a r t --> Cart c a r` t (rolled r vs. non rolled)
            #      Kite k a i t --> Kite K a t (southern dialect 'I')
            
            for p in change_phones:
                pron_ = pron.replace(p, phone_map[p])
                lex.append((w, pron_))

    for i in lex:
        print(u"{}\t{}".format(i[0], i[1]).encode("utf-8"))

if __name__ == "__main__":
    main()

