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
stage=0

. ./utils/parse_options.sh

if [ $stage -le 0 ]; then
  echo "stage 0: Setting up individual languages"
  ./local/setup_languages.sh --langs "${langs}" --recog "${recog}"
fi

if [ $stage -le 1 ]; then
  echo "stage 1: HMM-GMM training to get alignments"
  ./local/get_alignments.sh
fi
