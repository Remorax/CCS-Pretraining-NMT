function join_by { local d=${1-} f=${2-}; if shift 2; then printf %s "$f" "${@/#/$d}"; fi; }

set -e
# Variables to change
prefix="from-scratch"
train_langs=(fi_FI et_EE)
all_langs=(fi_FI et_EE en_XX)

# More or less fixed
base_dir="~/CAS"
datasets_dir="~/CAS/data"
FAIRSEQ=~/fairseq/fairseq_cli # Fairseq path
SPM=~/sentencepiece/build/src # Sentencepiece path

# Derived
data_dir="${base_dir}/data/${prefix}"
checkpoints_dir="${base_dir}/checkpoints/${prefix}"
results_dir="${base_dir}/results/${prefix}"
dict_dir="${data_dir}/dict"
spm_dir="${data_dir}/spm"
spm_corpus="${data_dir}/spm_corpus"
MODEL=${spm_dir}/${prefix}.model

TRAIN_SRC="${data_dir}/train.src"
TRAIN_TGT="${data_dir}/train.tgt"
DEV_SRC="${data_dir}/dev.src"
DEV_TGT="${data_dir}/dev.tgt"

rm -rf ${data_dir} ${checkpoints_dir} ${results_dir} ${dict_dir} ${spm_dir} 
mkdir -p ${data_dir} ${checkpoints_dir} ${results_dir} ${dict_dir} ${spm_dir}

rm -rf ${spm_corpus} ${TRAIN_SRC} ${TRAIN_TGT} ${DEV_SRC} ${DEV_TGT}
touch $spm_corpus ${TRAIN_SRC} ${TRAIN_TGT} ${DEV_SRC} ${DEV_TGT}

SRC_LANG="en_XX"
SRC_LANG_SHORT="${SRC_LANG:0:2}"

for TGT_LANG in "${train_langs[@]}"
do
    lang="${TGT_LANG:0:2}"
    SRC_LANG_CAPS="${SRC_LANG_SHORT^^}"
    TGT_LANG_CAPS="${lang^^}"

    # Variables to change
    PARALLEL_DIR="${datasets_dir}/en-${lang}-uralic"
    SRC_FILE="${PARALLEL_DIR}/train.${SRC_LANG}"
    REF_FILE="${PARALLEL_DIR}/train.${TGT_LANG}"
    
    cat ${SRC_FILE} ${REF_FILE} >> ${spm_corpus}
done
for SRC_LANG in "${all_langs[@]}"
do
    lang="${SRC_LANG:0:2}"
    SRC_LANG_CAPS="${SRC_LANG_SHORT^^}"

    # Variables to change
    PARALLEL_DIR="${datasets_dir}/mono-${lang}"
    SRC_FILE="${PARALLEL_DIR}/mono.${SRC_LANG}"
    
    cat ${SRC_FILE} >> ${spm_corpus}
done


shuf $spm_corpus > "${spm_corpus}.shuf"
rm $spm_corpus

echo "Training an spm model"

${SPM}/spm_train --input="${spm_corpus}.shuf" --train_extremely_large_corpus=true --model_prefix="${data_dir}/spm/${prefix}" --vocab_size=32000 --character_coverage=1.0  --model_type=unigram --shuffle_input_sentence=true

${SPM}/spm_encode --model="${MODEL}" < ${spm_corpus}.shuf > "${spm_corpus}.spm.src"

rm ${spm_corpus}.shuf

python ${FAIRSEQ}/preprocess.py \
        --source-lang src --target-lang src \
        --trainpref "${spm_corpus}.spm" \
        --destdir ${dict_dir} --only-source \
        --thresholdtgt 0 --thresholdsrc 0 \
        --dict-only --joined-dictionary  \
        --workers 32
rm "${spm_corpus}.spm.src"

for lang_large in "${all_langs[@]}"
do
    lang="${lang_large:0:2}"
    lang_caps="${lang^^}"
    echo -e "LANG_TOK_${lang_caps} 1" >> "${dict_dir}/dict.src.txt"
done

SRC_LANG="en_XX"
SRC_LANG_SHORT="${SRC_LANG:0:2}"
for TGT_LANG in "${train_langs[@]}"
do
    lang="${TGT_LANG:0:2}"
    SRC_LANG_CAPS="${SRC_LANG_SHORT^^}"
    TGT_LANG_CAPS="${lang^^}"

    # Variables to change
    PARALLEL_DIR="${datasets_dir}/en-${lang}-uralic"
    TRAIN_PREFIX="${PARALLEL_DIR}/train"
    DEV_PREFIX="${PARALLEL_DIR}/dev"
    TRAIN_TOK="${data_dir}/train"
    DEV_TOK="${data_dir}/dev"

    ${SPM}/spm_encode --model=${MODEL} < ${TRAIN_PREFIX}.${SRC_LANG} > ${TRAIN_TOK}.spm.${SRC_LANG_SHORT}
    ${SPM}/spm_encode --model=${MODEL} < ${TRAIN_PREFIX}.${TGT_LANG} > ${TRAIN_TOK}.spm.${lang}

    ${SPM}/spm_encode --model=${MODEL} < ${DEV_PREFIX}.${SRC_LANG} > ${DEV_TOK}.spm.${SRC_LANG_SHORT}
    ${SPM}/spm_encode --model=${MODEL} < ${DEV_PREFIX}.${TGT_LANG} > ${DEV_TOK}.spm.${lang}
    
    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' ${TRAIN_TOK}.spm.${SRC_LANG_SHORT} >> ${TRAIN_SRC}
    awk -v TGT=$TGT_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } ' ${TRAIN_TOK}.spm.${lang} >> ${TRAIN_TGT}

    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' ${TRAIN_TOK}.spm.${SRC_LANG_SHORT} >> ${TRAIN_TGT}
    awk -v TGT=$TGT_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } ' ${TRAIN_TOK}.spm.${lang} >> ${TRAIN_SRC}

    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' ${DEV_TOK}.spm.${SRC_LANG_SHORT} >> ${DEV_SRC}
    awk -v TGT=$TGT_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } ' ${DEV_TOK}.spm.${lang} >> ${DEV_TGT}

done

python ${FAIRSEQ}/preprocess.py \
    --source-lang src --target-lang tgt \
    --trainpref ${data_dir}/train \
    --validpref ${data_dir}/dev \
    --srcdict ${dict_dir}/dict.src.txt \
    --tgtdict ${dict_dir}/dict.src.txt \
    --destdir ${data_dir} \
    --thresholdtgt 0 --thresholdsrc 0 --seed 0 \
    --amp --workers 128

sbatch train.sh
