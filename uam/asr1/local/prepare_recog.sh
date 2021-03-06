#!/bin/bash

# Copyright 2018 Johns Hopkins University (Matthew Wiesner)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# This is not necessarily the top-level run.sh as it is in other directories.   see README.txt first.

. ./conf/lang.conf
. ./path.sh
. ./cmd.sh

. ./utils/parse_options.sh
if [ $# -ne 1 ]; then
  echo >&2 "Usage: ./local/prepare_data.sh [opts] <lang_id>"
  echo >&2 "       --extract-feats :  Extract plp features for dev10h directory"
  exit 1
fi

l=$1

#set -e           #Exit on non-zero return code from any command
#set -o pipefail  #Exit if any of the commands in the pipeline will
                 #return non-zero return code
#set -u           #Fail on an undefined variable


./local/check_tools.sh || exit 1

#Preparing train directories
if [ ! -f data/raw_train_data/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Subsetting the TRAIN set"
    echo ---------------------------------------------------------------------
    train_data_dir=train_data_dir_${l}_FLP
    train_data_list=train_data_list_${l}_FLP
    local/make_corpus_subset.sh "${!train_data_dir}" "${!train_data_list}" ./data/raw_train_data
    train_data_dir=`utils/make_absolute.sh ./data/raw_train_data`
    touch data/raw_train_data/.done
fi

nj_max=`cat ${!train_data_list} | wc -l`
if [[ "$nj_max" -lt "$train_nj" ]] ; then
    echo "The maximum reasonable number of jobs is $nj_max (you have $train_nj)! (The training and decoding process has file-granularity)"
    exit 1;
    train_nj=$nj_max
fi
train_data_dir=`utils/make_absolute.sh ./data/raw_train_data`


#Preparing dev10 directories
if [ ! -f data/raw_dev10h_data/.done ]; then
    echo ---------------------------------------------------------------------
    echo "Subsetting the Dev set"
    echo ---------------------------------------------------------------------
    dev10h_data_dir=dev10h_data_dir_${l}
    dev10h_data_list=dev10h_data_list_${l}
    local/make_corpus_subset.sh "${!dev10h_data_dir}" "${!dev10h_data_list}" ./data/raw_dev10h_data
    dev10h_data_dir=`utils/make_absolute.sh ./data/raw_dev10h_data`
    touch data/raw_dev10h_data/.done
fi
nj_max=`cat ${!dev10h_data_list} | wc -l`
if [[ "$nj_max" -lt "$train_nj" ]] ; then
    echo "The maximum reasonable number of jobs is $nj_max (you have $train_nj)! (The training and decoding process has file-granularity)"
    exit 1;
    train_nj=$nj_max
fi
dev10h_data_dir=`utils/make_absolute.sh ./data/raw_dev10h_data`

mkdir -p data/dict_flp
# We always assume we have the big lexicon in this case
lexicon_file=lexicon_file_${l}_FLP
lexiconFlags=lexiconFlags_${l}
if [[ ! -f $lexicon || $lexicon -ot "${!lexicon_file}" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing lexicon in data/local on" `date`
  echo ---------------------------------------------------------------------
  local/prepare_lexicon.pl ${!lexiconFlags} ${!lexicon_file} data/dict_flp
fi


if [[ ! -f data/dev10h.pem/wav.scp || data/dev10h.pem/wav.scp -ot "$dev10h_data_dir" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing acoustic dev10h lists in data/dev10h.pem on" `date`
  echo ---------------------------------------------------------------------
  mkdir -p data/dev10h.pem
  local/prepare_acoustic_training_data.pl \
    --vocab data/dict_flp/lexicon.txt --fragmentMarkers \-\*\~ \
    $dev10h_data_dir data/dev10h.pem > data/dev10h.pem/skipped_utts.log
  
  ./local/prepare_stm.pl --fragmentMarkers \-\*\~ data/dev10h.pem
fi


if [[ ! -f data/train/wav.scp || data/train/wav.scp -ot "$train_data_dir" ]]; then
  echo ---------------------------------------------------------------------
  echo "Preparing acoustic train lists in data/train on" `date`
  echo ---------------------------------------------------------------------
  mkdir -p data/train
  local/prepare_acoustic_training_data.pl \
    --vocab data/dict_flp/lexicon.txt --fragmentMarkers \-\*\~ \
    $train_data_dir data/train > data/train/skipped_utts.log
fi


# Make Features
dev_dir=data/dev10h.pem
steps/make_plp_pitch.sh --nj 32 --cmd "$train_cmd" ${dev_dir} exp/make_plp_pitch/dev10h.pem plp
steps/compute_cmvn_stats.sh ${dev_dir} exp/make_plp_pitch/dev10h.pem plp
utils/fix_data_dir.sh data/dev10h.pem

