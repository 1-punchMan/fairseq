DATA_DIR="/home/zchen/fairseq/data/processed/wiki_en-zh/"
CHECKPATH="/home/zchen/fairseq/mBART/checkpoints/pretraining/2"
LOG=$CHECKPATH/log
LANGS="en,zh"

export CUDA_VISIBLE_DEVICES=1

if [ -d $CHECKPATH ]; then
  echo "The saving directory exists!"
  exit
fi

mkdir -p $CHECKPATH

fairseq-train --task multilingual_denoising \
  $DATA_DIR \
  --save-dir $CHECKPATH \
  --mask 0.3 --mask-random 0.1 --rotate 0 --poisson-lambda 3.5 --permute-sentences 1 \
  --mask-length span-poisson --replace-length 1 \
  --max-source-positions 1024 --max-target-positions 1024 \
  --multilang-sampling-alpha 0.7 --add-lang-token --langs $LANGS \
  --arch mbart_base --share-decoder-input-output-embed --share-all-embeddings --layernorm-embedding \
  --dropout 0 \
  --optimizer adam --adam-eps 1e-06 --weight-decay 0.01 --clip-norm 0.1 \
  --lr 0.0003 --lr-scheduler polynomial_decay --warmup-updates 10000 --total-num-update 500000 \
  --tokens-per-sample 512 --sample-break-mode eos \
  --max-tokens 3000 --max-sentences 32 \
  --fp16 \
  --validate-interval 10000000000 --no-epoch-checkpoints --no-last-checkpoints \
  --save-interval-updates 3000 --keep-interval-updates 1 --patience 25 \
  --log-format simple --log-file $LOG