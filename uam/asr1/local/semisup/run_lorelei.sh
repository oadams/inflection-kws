#!/bin/bash

# Copyright 2019 Matthew Wiesner
# Apache 2.0

# This script is a semi-supervised recipe using a seed universal acoustic model
# with unsupervised data (provided by LORELEI) to perform semi-supervised
# training with LF-MMI
# (http://www.danielpovey.com/files/2018_icassp_semisupervised_mmi.pdf)
#

. ./path.sh
. ./cmd.sh

target_data=
target_name=semisup
target_lang=
src_model=
src_tree=
src_ivector_extractor=
odir=

. ./utils/parse_options.sh

target_lang_name=`basename ${target_lang}`
graphdir=${src_model}/graph_semisup_${target_lang_name}

###############################################################################
#                  Make the graph for semisupervised decoding                 
###############################################################################
if [ ! -f $graphdir/HCLG.fst ]; then
  utils/mkgraph.sh --self-loop-scale 1.0 $target_lang $src_model $graphdir
fi


###############################################################################
#                         Prepare target data features                
###############################################################################
./utils/data/perturb_data_dir_speed_3way.sh ${target_data} ${target_data}_sp_hires
./utils/data/perturb_data_dir_volume.sh ${target_data}_sp_hires

./steps/make_mfcc_pitch.sh --nj ${nj} \
                           --cmd "$train_cmd" \
                           --mfcc-config conf/mfcc_hires.conf \
                           ${target_data}_sp_hires

./utils/copy_data_dir.sh ${target_data}_sp_hires ${target_data}_sp_hires_nopitch
./utils/data/limit_feature_dim.sh 0:39 ${target_data}_sp_hires_nopitch

utils/data/modify_speaker_info.sh --utts-per-spk-max 2 \
    ${target_data}_sp_hires_nopitch ${target_data}_sp_hires_nopitch_max2

./steps/online/nnet2/extract_ivectors_online.sh --cmd "$train_cmd" --nj ${nj} \
  ${target_data}_sp_hires_nopitch_max2 ${src_ivector_extractor} ${target_data}_ivectors

###############################################################################
#                         Decode unsupervised data                
###############################################################################
./steps/nnet3/decode_semisup.sh --num-threads 4 --nj ${nj} --cmd "$decode_cmd" \
                                --acwt 1.0 --post-decode-acwt 10.0 \
                                --write-compact false --skip-scoring true \
                                --online-ivector-dir ${target_data}_ivectors \
                                --scoring-opts "--min-lmwt 10 --max-lmwt 10" \
                                --word-determinize false \
                                ${graph_dir} ${target_data}_sp_hires ${src_model}/decode_${target_name}


###############################################################################
#                         Get best alignment from lattice posterior                
###############################################################################
./steps/best_path_weights.sh --cmd "$train_cmd" --acwt 0.1 \
  ${target_data}_sp_hires ${src_model}/decode_${target_name} ${src_model}/best_path_${target_name}

frame_subsampling_factor=3
cmvn_opts=$(cat ${src_model}/cmvn_opts) || exit 1

./steps/nnet3/chain/make_weighted_den_fst.sh --num-repeats ${lm_weights} \
  --cmd "$train_cmd" ${src_tree} ${src_model}/best_path_${target_name} ${odir}
 
 
 


