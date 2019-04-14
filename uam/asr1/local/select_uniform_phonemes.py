#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright 2018 Johns Hopkins University (Author: Matthew Wiesner)
# Apache 2.0

from __future__ import print_function
import codecs
import sys
import os
import argparse
import random


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("text", help="Kaldi text file", type=str)
    parser.add_argument("lexicon", help="Lexicon", type=str)
    parser.add_argument("N", help="Number of utts to select", type=int)
    args = parser.parse_args()

    lexicon = {}
    phones = set()
    with codecs.open(args.lexicon, 'r', encoding='utf-8') as f:
        for l in f:
            word, pron = l.strip().split(1, None)
            pron = pron.split()
            phones = phones.union(pron)
            lexicon[word] = pron

    utts = {}
    with codecs.open(args.text, 'r', encoding='utf-8') as f:
        for l in f:
            utt_id, text = l.strip().split(1, None)
            pron_set = set()
            for w in text:
                try:
                    pron_set = pron_set.union(lexicon[w])
                except KeyError:
                    pass
            utts[utt_id] = pron_set


    utts_per_phone = float(args.N) / len(phones)
    selected = {}
    for utt in random.shuffle(utts.keys()):
        lang_id = utt.split('_')[0]
        for p in phones:
            if p in utts[utt]:
                selected[lang_id].append(
                
            
        

if __name__ == "__main__":
    main()
