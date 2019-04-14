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
echo "Languagues: ${langs}"

# Basic directory prep
for l in ${recog2}; do
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

# Prepare recog data
for l in ${recog2}; do
  cd data/${l}
  l=${l%%_test}
  ./local/prepare_recog.sh ${l}
  cd ${cwd}
done
