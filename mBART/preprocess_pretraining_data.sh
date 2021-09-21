# Ensure the output directory exists
PROCESSED_XLM_DATA=/home/zchen/CLTS/data/processed/XLM_en_zh/50k-chiamin
DATA_DIR=monolingual_data/fairseq_processed
mkdir -p "$DATA_DIR"

for lg in en zh
do

  fairseq-preprocess \
  --srcdict $PROCESSED_XLM_DATA/vocab.en-zh \
  --only-source \
  --trainpref $PROCESSED_XLM_DATA/train \
  --validpref $PROCESSED_XLM_DATA/valid \
  --testpref $PROCESSED_XLM_DATA/test \
  --destdir $DATA_DIR \
  --workers 20 \
  --source-lang $lg

  # Since we only have a source language, the output file has a None for the
  # target language. Remove this

  for stage in train test valid
  do

    # mv "$stage.$lg.bin" "$DATA_DIR/"
    # mv "$stage.$lg.idx" "$DATA_DIR/"

  done

done
