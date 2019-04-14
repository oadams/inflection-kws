#!/bin/bash

. ./path.sh
. ./cmd.sh

recog="107 201 307 404"
my_nj=10
stage=0

. ./utils/parse_options.sh

model=$1

# We're going to assume all recog directories have the suffix "test"
recog2=""
for l in ${recog}; do
  recog2="${l}_test ${recog2}"
done
recog2=${recog2%% }


if [ $stage -le 0 ]; then
  for l in ${recog2}; do
    (
      ./utils/mkgraph.sh --self-loop-scale 1.0 \
        data/${l}/data/lang_universal \
        exp/${model} \
        exp/${model}/graph_${l} | tee exp/${model}/mkgraph.${l}.log
    ) &
  done
  wait
fi

if [ $stage -le 1 ]; then
  decode_extra_opts=(--num-threads 6 --parallel-opts "--num-threads 6 --mem 4G")
  for l in ${recog2}; do
    (
      if [[ $model == "tri1" ]]; then
        echo "Decoding tri1"
        steps/decode_si.sh --nj $my_nj --cmd "$decode_cmd" exp/${model}/graph_${l} data/${l}/data/dev10h.pem exp/${model}/decode_${l} | tee exp/${model}/decode_${l}/decode.log
      elif [[ $model == "tri2" ]]; then
        echo "Decoding tri2"
      elif [[ $model == "tri3" ]]; then
        echo "Decoding tri3"
      elif [[ $model == "tri4" ]]; then
        echo "Decoding tri4"
      elif [[ $model == "tri5" ]]; then
        echo "Decoding tri5"
        steps/decode_fmllr_extra.sh --skip-scoring true --beam 10 --lattice-beam 4\
          --nj $my_nj --cmd "$decode_cmd" "${decode_extra_opts[@]}"\
          exp/${model}/graph_${l} data/${l}/data/dev10h.pem exp/${model}/decode_${l} |tee exp/${model}/decode_${l}/decode.log
      fi
    ) &
  done
  wait
fi


if [ $stage -le 2 ]; then
  for l in ${recog2}; do
    (   
      local/lattice_to_ctm.sh --cmd "$decode_cmd" --word-ins-penalty 0.5 \
        --min-lmwt 7 --max-lmwt 17 --resolve-overlaps false \
        data/${l}/data/dev10h.pem data/${l}/data/lang_universal exp/${model}/decode_${l}
  
      local/score_stm.sh --cmd "$decode_cmd" --cer 0 \
        --min-lmwt 7 --max-lmwt 17 \
        data/${l}/data/dev10h.pem data/${l}/data/lang_universal exp/${model}/decode_${l}
    ) &
  done
fi
  
