# apply BPE to the tokenized data
FASTBPE=/home/zchen/CLTS/tools/fastBPE/fast
CODES="/home/zchen/CLTS/data/processed/XLM_en_zh/50k-chiamin/codes"
DATA="/home/zchen/fairseq/data/clts/zh-en"
SRC=zh
TGT=en

for split in train valid test; do
    mv $DATA/$split-doc.tok $DATA/$split.$SRC.tok
    mv $DATA/$split-sum.tok $DATA/$split.$TGT.tok
done

for lg in $SRC $TGT; do
    for split in train valid test; do
        $FASTBPE applybpe $DATA/$split.$lg $DATA/$split.$lg.tok $CODES
    done
done

# fairseq-preprocess
OUT_DIR=/home/zchen/fairseq/data/processed/clts/zh-en

mkdir -p "$OUT_DIR"

DICT="/home/zchen/fairseq/data/processed/wiki_en-zh/dict.txt"
fairseq-preprocess \
  --source-lang ${SRC} \
  --target-lang ${TGT} \
  --trainpref ${DATA}/train \
  --validpref ${DATA}/valid \
  --testpref ${DATA}/test \
  --destdir ${OUT_DIR} \
  --thresholdtgt 0 \
  --thresholdsrc 0 \
  --srcdict ${DICT} \
  --tgtdict ${DICT} \
  --workers 70