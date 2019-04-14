#!/bin/bash

. ./path.sh

lang="ru"

. ./utils/parse_options.sh
if [ $# -ne 1 ]; then
  echo >&2 "Usage: ./local/prepare_voxforge.sh <dir>"
  exit 1;
fi

dir=$1

# Back up text
cp ${dir}/text ${dir}/text.bk

# Convert text to lower case to match lexicon
paste -d' ' <(cut -d' ' -f1 ${dir}/text.bk) \
            <(cut -d' ' -f2- ${dir}/text.bk | LC_ALL= sed 's/./\L&/g') \
            > ${dir}/text

grep 'anonymous' ${dir}/text | awk '{print $1}' > ${dir}/anon.map.tmp
paste -d' ' ${dir}/anon.map.tmp <(grep -o "anonymous-.*-${lang}" ${dir}/anon.map.tmp) > ${dir}/anon.map

grep -v 'anonymous' ${dir}/text | awk '{print $1}' > ${dir}/nonanon.map.tmp
paste -d' ' ${dir}/nonanon.map.tmp <(grep -o '^[^-]*' ${dir}/nonanon.map.tmp) > ${dir}/nonanon.map 

awk '{print $1" "$1}' ${dir}/text |\
utils/apply_map.pl -f 2- --permissive ${dir}/nonanon.map 2>/dev/null |\
utils/apply_map.pl -f 2- --permissive ${dir}/anon.map 2>/dev/null > ${dir}/utt2spk

./utils/utt2spk_to_spk2utt.pl ${dir}/utt2spk > ${dir}/spk2utt



exit
