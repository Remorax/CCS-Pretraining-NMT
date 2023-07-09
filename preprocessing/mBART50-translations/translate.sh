set -e 

SPM=~/sentencepiece/build/src # Sentencepiece path
FAIRSEQ=~/fairseq/fairseq_cli # Fairseq path
max_tokens_test=4000    
WORLD_SIZE=3

src="${src_long:0:2}"
tgt="${tgt_long:0:2}"

source ~/.bashrc
source activate fairseq-mbart

lang_pairs="${src_long}-${tgt_long}"
    
PREFIX=${lang_pairs}_${OUTPUT_PREFIX}
PREPROCESSED=${DATASET_DIR}/preprocessed/cas/${PREFIX}
POSTPROCESSED=${DATASET_DIR}/postprocessed/cas/${PREFIX}

rm -rf  ${POSTPROCESSED} ${PREPROCESSED}
mkdir -p ${PREPROCESSED} ${POSTPROCESSED}
# cp ${DATASET_DIR}/${TRAIN_PREFIX}.${src_long} ${PREPROCESSED}/${TRAIN_PREFIX}.${tgt_long}
${SPM} --model=${SPM_MODEL} < ${DATASET_DIR}/${TRAIN_PREFIX}.${src_long} > ${PREPROCESSED}/${TRAIN_PREFIX}.spm.${src_long}
cp ${PREPROCESSED}/${TRAIN_PREFIX}.spm.${src_long} ${PREPROCESSED}/${TRAIN_PREFIX}.spm.${tgt_long}
echo "Done"

python ${FAIRSEQ}/preprocess.py \
    --source-lang ${src_long} --target-lang ${tgt_long} \
    --testpref ${PREPROCESSED}/${TRAIN_PREFIX}.spm \
    --destdir ${POSTPROCESSED} \
    --thresholdsrc 0 --thresholdtgt 0 \
    --srcdict ${DICT_SRC} --tgtdict ${DICT_TGT} \
    --workers 4

rm -rf ${PREPROCESSED}/results/*

for i in {0..2};
do
    mkdir -p ${PREPROCESSED}/results/shard_id${i}/
    CUDA_VISIBLE_DEVICES=${i} python ${FAIRSEQ}/generate.py ${POSTPROCESSED} \
            --max-tokens ${max_tokens_test} \
            --path ${MODEL} --lang-dict ${lang_list} \
            --fixed-dictionary ${DICT_SRC} \
            --source-lang ${src_long} --target-lang ${tgt_long} \
            --remove-bpe 'sentencepiece' --beam 5 \
            --task translation_multi_simple_epoch \
            --lang-pairs "${src_long}-${tgt_long}" \
            --decoder-langtok --encoder-langtok src \
            --gen-subset test --dataset-impl mmap \
            --distributed-world-size ${WORLD_SIZE} --distributed-rank ${i} \
            --results-path ${PREPROCESSED}/results/shard_id${i} &
done
wait

echo "Done Generation. Postprocessing..."
rm -f "${PREPROCESSED}/results/generate-test-all.txt"
touch "${PREPROCESSED}/results/generate-test-all.txt"

for i in {0..2};
do
        cat "${PREPROCESSED}/results/shard_id${i}/generate-test.txt" >> "${PREPROCESSED}/results/generate-test-all.txt"
done
rm -f "${PREPROCESSED}/translated_${OUTPUT_PREFIX}.detok.${tgt_long}"
cat "${PREPROCESSED}/results/generate-test-all.txt" | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' > "${PREPROCESSED}/translated_${OUTPUT_PREFIX}.detok.${tgt_long}"

echo "All Done."

cd ../awesome-align
SRC_PREFIX="${DATASET_DIR}/${TRAIN_PREFIX}"
TGT_PREFIX="${PREPROCESSED}/translated_${OUTPUT_PREFIX}.detok"
ALIGNER_MODEL_DIR="${FT_ALIGN_DIR}"
OUTPUT_PREFIX=".ft"
sbatch --export=SRC=$src_long,TGT=$tgt_long,SRC_PREFIX=$SRC_PREFIX,TGT_PREFIX=$TGT_PREFIX,MODEL_NAME_OR_PATH=$ALIGNER_MODEL_DIR,OUTPUT_PREFIX=$OUTPUT_PREFIX aa-align.sh