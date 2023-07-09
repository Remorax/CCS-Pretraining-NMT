SCRIPTS_DIR=~/scripts/
SRC_SHORT="${SRC:0:2}"
TGT_SHORT="${TGT:0:2}"

TOK_PREFIX="${TGT_PREFIX}${OUTPUT_PREFIX}.tok"
JOINED_FILE="${TGT_PREFIX}${OUTPUT_PREFIX}.joined"

OUTPUT_FILE="${TGT_PREFIX}${OUTPUT_PREFIX}.alignments"

cat ${SRC_PREFIX}.${SRC} | perl ${SCRIPTS_DIR}/moses_tokenizer.pl -a -q -l ${SRC_SHORT} -no-escape -threads 42 > ${TOK_PREFIX}.${SRC}
cat ${TGT_PREFIX}.${TGT} | perl ${SCRIPTS_DIR}/moses_tokenizer.pl -a -q -l ${TGT_SHORT} -no-escape -threads 42 > ${TOK_PREFIX}.${TGT}

:|paste -d ' ||| ' ${TOK_PREFIX}.${SRC} - - - - ${TOK_PREFIX}.${TGT} > $JOINED_FILE
rm -rf $OUTPUT_FILE
awesome-align \
    --output_file=$OUTPUT_FILE \
    --model_name_or_path=$MODEL_NAME_OR_PATH \
    --data_file=$JOINED_FILE \
    --extraction 'softmax' \
    --batch_size 512

