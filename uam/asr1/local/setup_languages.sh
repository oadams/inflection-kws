#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

set -e
set -o pipefail
. ./path.sh
. ./cmd.sh
. ./conf/lang.conf

langs="101 102 103 104 105 106 202 203 204 205 206 207 301 302 303 304 305 306 401 402 403"
recog="107 201 307 404"

. ./utils/parse_options.sh

# We're going to assume all recog directories have the suffix "test"
recog2=""
for l in ${recog}; do
  recog2="${l}_test ${recog2}"
done
recog2=${recog2%% }

# Save top-level directory
cwd=$(utils/make_absolute.sh `pwd`)
echo "Stage 0: Setup Language Specific Directories"

echo " --------------------------------------------"
echo "Languages: ${langs}"

# Basic directory prep
for l in ${langs} ${recog2}; do


  [ -d data/${l} ] || mkdir -p data/${l}
  cd data/${l}

  ln -sf ${cwd}/local .
  for f in ${cwd}/{utils,steps,conf}; do
    link=`make_absolute.sh $f`
    ln -sf $link .
  done

  cp ${cwd}/cmd.sh .
  cp ${cwd}/path.sh .
  sed -i 's/`pwd`\/\.\.\/\.\.\/\.\.\//`pwd`\/\.\.\/\.\.\/\.\.\/\.\.\/\.\.\//g' path.sh
  
  cd ${cwd}
done

# Prepare language specific data
for l in ${langs}; do
  (
    cd data/${l}
    ./local/prepare_data.sh --extract-feats true ${l}
    ./local/prepare_universal_dict.sh --dict data/dict_universal ${l}
    cd ${cwd}
  ) &
done
wait

# Combine all language specific training directories and generate a single
# lang directory by combining all language specific dictionaries
train_dirs=""
train_sub1=""
train_sub2=""
train_sub3=""
dict_dirs=""
for l in ${langs}; do
  train_dirs="data/${l}/data/train_${l} ${train_dirs}"
  train_sub1="data/${l}/data/train_${l}_sub1 ${train_sub1}"
  train_sub2="data/${l}/data/train_${l}_sub2 ${train_sub2}"
  train_sub3="data/${l}/data/train_${l}_sub3 ${train_sub3}"
  dict_dirs="data/${l}/data/dict_universal ${dict_dirs}"
done

./utils/combine_data.sh data/train ${train_dirs}
./utils/combine_data.sh data/train_sub1 ${train_sub1}
./utils/combine_data.sh data/train_sub2 ${train_sub2}
./utils/combine_data.sh data/train_sub3 ${train_sub3}

./local/combine_lexicons.sh data/dict_universal ${dict_dirs}

# Prepare lang directory
./utils/prepare_lang.sh --share-silence-phones true \
  data/dict_universal "<unk>" data/dict_universal/tmp.lang data/lang_universal

#if [ ! -f data/train_sub3/.done ]; then
#  echo ---------------------------------------------------------------------
#  echo "Subsetting monophone training data in data/train_sub[123] on" `date`
#  echo ---------------------------------------------------------------------
#  numutt=`cat data/train/feats.scp | wc -l`;
#  # We want to ensure coverage of all phonemes so we enforce
#  # selection explicitly from all "speakers" = lang+speaker
#  #./utils/subset_data_dir.sh --per-speaker data/train 5 data/train_sub1 
#  #./utils/subset_data_dir.sh --per-speaker data/train 15 data/train_sub2
#  #./utils/subset_data_dir.sh --per-speaker data/train 45 data/train_sub3 
#  ./utils/subset_data_dir.sh data/train 10000 data/train_sub1
#  ./utils/subset_data_dir.sh data/train 40000 data/train_sub2
#  ./utils/subset_data_dir.sh data/train 10000 data/train_sub3
#  touch data/train_sub3/.done
#fi

# Prepare recog data
for l in ${recog2}; do
    echo "Recog language: $l"
  (
    cd data/${l}
    l=${l%%_test}
    ./local/prepare_recog.sh ${l}
    ./local/prepare_universal_dict.sh --src data/dict_flp --dict data/dict_universal ${l}_test
    
    ./utils/prepare_lang.sh --share-silence-phones true \
                            --phone-symbol-table ${cwd}/data/lang_universal/phones.txt\
                            data/dict_universal "<unk>" data/dict_universal/tmp.lang data/lang_universal
    ###########################################################################
    # Train the LM For the Decoding Language
    ###########################################################################
    ./local/train_lms_srilm.sh --oov-symbol "<unk>" \
                             --train-text data/train/text \
                             --words-file data/lang_universal/words.txt \
                             data data/srilm
    
    ./local/arpa2G.sh data/srilm/lm.gz data/lang_universal data/lang_universal
    
    
    cd ${cwd}
  )
done
