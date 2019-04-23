export KALDI_ROOT=/export/b13/oadams/kws/kaldi
[ -f $KALDI_ROOT/tools/env.sh ] && . $KALDI_ROOT/tools/env.sh

#[ ! -f /export/babel/data/software/env.sh ] && echo >&2 "The file /export/babel/data/software/env.sh is not present -> Exit!" && exit 1
#. /export/babel/data/software/env.sh

export PATH=$PWD/utils/:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/sph2pipe_v2.5/:$PWD:$PATH

[ ! -f $KALDI_ROOT/tools/config/common_path.sh ] && echo >&2 "The standard file $KALDI_ROOT/tools/config/common_path.sh is not present -> Exit!" && exit 1
. $KALDI_ROOT/tools/config/common_path.sh

export LC_ALL=C

cpanm --local-lib=~/perl5 local::lib &> /dev/null && eval $(perl -I ~/perl5/lib/perl5/ -Mlocal::lib)
export PATH=$PATH:/export/b13/oadams/kws/tools/F4DE-3.5.0/bin
