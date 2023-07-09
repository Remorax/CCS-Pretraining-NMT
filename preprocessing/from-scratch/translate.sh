set -e 

SPM=~/sentencepiece/build/src
base_dir=~/CAS
FAIRSEQ=~/fairseq/fairseq_cli
max_tokens_test=4000    
WORLD_SIZE=3

src="${src_long:0:2}"
tgt="${tgt_long:0:2}"


lang_pairs="${src_long}-${tgt_long}"
    
PREFIX=${lang_pairs}_${OUTPUT_PREFIX}
PREPROCESSED=${DATASET_DIR}/preprocessed/cas/${PREFIX}
POSTPROCESSED=${DATASET_DIR}/postprocessed/cas/${PREFIX}

rm -rf  ${POSTPROCESSED} ${PREPROCESSED}
mkdir -p ${PREPROCESSED} ${POSTPROCESSED}
# cp ${DATASET_DIR}/${TRAIN_PREFIX}.${src_long} ${PREPROCESSED}/${TRAIN_PREFIX}.${tgt_long}
${SPM}/spm_encode --model=${SPM_MODEL} < ${DATASET_DIR}/${TRAIN_PREFIX}.${src_long} > ${PREPROCESSED}/${TRAIN_PREFIX}.spm.${src_long}
cp ${PREPROCESSED}/${TRAIN_PREFIX}.spm.${src_long} ${PREPROCESSED}/${TRAIN_PREFIX}.spm.${tgt_long}
echo "Done"

python ${FAIRSEQ}/preprocess.py \
    --source-lang ${src_long} --target-lang ${tgt_long} \
    --testpref ${PREPROCESSED}/${TRAIN_PREFIX}.spm \
    --destdir ${POSTPROCESSED} \
    --thresholdsrc 0 --thresholdtgt 0 \
    --srcdict ${DICT_SRC} --tgtdict ${DICT_TGT} \
    --workers 42

rm -rf ${PREPROCESSED}/results/*

for i in {0..2};
do
    mkdir -p ${PREPROCESSED}/results/shard_id${i}/
    CUDA_VISIBLE_DEVICES=${i} python ${FAIRSEQ}/generate.py ${POSTPROCESSED} \
            --max-tokens ${max_tokens_test} \
            --user-dir ${base_dir}/mcolt \
            --max-source-positions ${max_tokens_test} \
            --max-target-positions ${max_tokens_test} \
            --path ${MODEL} \
            --source-lang ${src_long} --target-lang ${tgt_long} \
            --beam 5 --task translation_w_langtok \
            --lang-prefix-tok "LANG_TOK_"`echo "${tgt} " | tr '[a-z]' '[A-Z]'` \
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

cat "${PREPROCESSED}/results/generate-test-all.txt" | grep -P '^H-'  | cut -c 3- | sort -n -k 1 | awk -F "\t" '{print $NF}' | cut -d " " -f2- > "${PREPROCESSED}/results/output.postproc.final.txt"
${SPM}/spm_decode --model=${SPM_MODEL} < "${PREPROCESSED}/results/output.postproc.final.txt" > "${PREPROCESSED}/translated_${OUTPUT_PREFIX}.detok.${tgt_long}"

SRC_PREFIX="${DATASET_DIR}/${TRAIN_PREFIX}"
TGT_PREFIX="${PREPROCESSED}/translated_${OUTPUT_PREFIX}.detok"
ALIGNER_MODEL_DIR="${FT_ALIGN_DIR}"
OUTPUT_PREFIX=".ft"
cd ../awesome-align
sbatch --export=SRC=$src_long,TGT=$tgt_long,SRC_PREFIX=$SRC_PREFIX,TGT_PREFIX=$TGT_PREFIX,MODEL_NAME_OR_PATH=$ALIGNER_MODEL_DIR,OUTPUT_PREFIX=$OUTPUT_PREFIX aa-align.sh

