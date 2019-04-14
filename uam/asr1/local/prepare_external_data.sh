#!/bin/bash

. ./path.sh
. ./cmd.sh

nbest=2
upper=false
. ./utils/parse_options.sh

if [ $# -ne 3 ]; then
  echo "Usage: ./local/prepare_external_data.sh <lex> <datadir> <odir>"
  exit 1;
fi

lexicon=$1
data=$2
odir=$3

data_id=`basename ${data}`

cut -d' ' -f2- ${data}/text | ngram-count -order 1 -text - -sort -write - | awk '{print $1}' > ${data}/vocab

mkdir -p ${odir}

if $upper; then
  comm -12 <(awk '{print $1}' ${lexicon} | LC_ALL= sed 's/./\U&/g' | LC_ALL=C sort) \
           <(grep -v '</*s>' ${data}/vocab | sort) > ${odir}/ivs
  comm -13 <(awk '{print $1}' ${lexicon} | LC_ALL= sed 's/./\U&/g' | LC_ALL=C sort) \
           <(grep -v '</*s>' ${data}/vocab | sort) > ${odir}/oovs
else
  comm -12 <(awk '{print $1}' ${lexicon} | LC_ALL=C sort) \
           <(grep -v '</*s>' ${data}/vocab | sort) > ${odir}/ivs
  comm -13 <(awk '{print $1}' ${lexicon} | LC_ALL=C sort) \
           <(grep -v '</*s>' ${data}/vocab | sort) > ${odir}/oovs
fi

../../process_set0/set0/local/train_g2p.sh ${lexicon} ${odir}/g2p

../../process_set0/set0/local/apply_g2p.sh --nbest ${nbest} ${odir}/oovs ${odir}/g2p ${odir}/g2p/oovs

cat ${odir}/g2p/oovs/lexicon_out.* | cut -f1,3 | grep -v '^<.*>\s' |\
cat - <(awk '(NR==FNR){a[$1]=1; next} ($1 in a){print $0}' ${odir}/ivs ${lexicon}) \
 > ${odir}/lexicon.raw

grep -v '</*s>' ${data}/vocab | awk -v var=${data_id} '{print $1" "$1"_"var}' > ${data}/vocab.map

./local/prepare_lexicon.pl --oov "<unk>" ${odir}/lexicon.raw ${odir}

mv ${odir}/lexicon.txt ${odir}/lexicon.bk
mv ${data}/text ${data}/text.bk

cat ${odir}/lexicon.bk | ./utils/apply_map.pl -f 1 --permissive ${data}/vocab.map 2>/dev/null |\
sed 's/ /\t/;s/\s*$//' > ${odir}/lexicon.txt

cat ${data}/text.bk | ./utils/apply_map.pl -f 2- --permissive ${data}/vocab.map 2>/dev/null > ${data}/text


