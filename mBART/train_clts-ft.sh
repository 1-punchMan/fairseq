DATAPATH="/home/zchen/fairseq/data/processed/clts/zh-en/"
PRETRAIN="/home/zchen/fairseq/mBART/checkpoints/pretraining/1/checkpoint_best.pt"
CHECKPATH="/home/zchen/fairseq/mBART/checkpoints/clts-ft-zh-en/1"
LOG=$CHECKPATH/log
langs=en,zh # These should match the langs from pretraining (and be in the same order).

export CUDA_VISIBLE_DEVICES=1

if [ -d $CHECKPATH ]; then
  echo "The saving directory exists!"
  exit
fi

mkdir -p $CHECKPATH

fairseq-train $DATAPATH \
  --save-dir $CHECKPATH \
  --encoder-normalize-before --decoder-normalize-before \
  --arch mbart_base --layernorm-embedding \
  --task translation_from_pretrained_bart \
  --source-lang zh --target-lang en \
  --criterion label_smoothed_cross_entropy --label-smoothing 0.2 \
  --optimizer adam --adam-eps 1e-06 --adam-betas '(0.9, 0.98)' \
  --lr-scheduler polynomial_decay --lr 3e-05 --warmup-updates 2500 --total-num-update 40000 \
  --dropout 0.3 --attention-dropout 0.1 --weight-decay 0.0 \
  --max-tokens 3000 --max-sentences 32 \
  --fp16 \
  --validate-interval 10000000000 --no-epoch-checkpoints --no-last-checkpoints \
  --save-interval-updates 600 --keep-interval-updates 1 --patience 10 \
  --log-format simple --log-file $LOG \
  --restore-file $PRETRAIN \
  --reset-optimizer --reset-meters --reset-dataloader --reset-lr-scheduler \
  --langs $langs \