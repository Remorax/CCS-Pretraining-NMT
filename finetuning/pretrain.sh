ulimit -n 6000
set -e

function join_by { local d=${1-} f=${2-}; if shift 2; then printf %s "$f" "${@/#/$d}"; fi; }

# Variables to change
prefix="sample-pretrain"
test_langs=(fi_FI et_EE)

# More or less fixed
base_dir="~/CAS" # Replace with the base directory of the project 
datasets_dir="~/CAS/data" # Replace with directory containing all datasets
SPM=~/sentencepiece/build/src # Sentencepiece path
FAIRSEQ=~/fairseq/fairseq_cli # Fairseq path
training_config="${base_dir}/configs/cas.yml"

# Derived
data_dir="${base_dir}/data/${prefix}"
checkpoints_dir="${base_dir}/checkpoints/${prefix}"
dict_dir="${data_dir}/dict"
results_dir="${base_dir}/results/${prefix}"
spm_dir="${data_dir}/spm"

if [[ -f "${results_dir}/testing.done" ]];
then
    exit 1;
fi

source ${base_dir}/scripts/load_config.sh ${training_config} ${base_dir}

mkdir -p ${checkpoints_dir} ${results_dir}

if [[ ! -f "${results_dir}/training.done" ]];
then
    python ${FAIRSEQ}/train.py ${data_dir}/mono \
        --user-dir ${base_dir}/mcolt \
        --save-dir ${checkpoints_dir} \
        --only_parallel \
        ${options} --do_shuf --patience 10 \
        --ddp-backend no_c10d 1>&2
fi

touch "${results_dir}/training.done"

SPM_MODEL=${spm_dir}/${prefix}.model
MODEL="${checkpoints_dir}/checkpoint_best.pt"

# Preprocessing for launching MLFT script
new_prefix="sample-mlft"

data_dir="${base_dir}/data/${new_prefix}"
checkpoints_dir="${base_dir}/checkpoints/${new_prefix}"
dict_dir="${data_dir}/dict"
results_dir="${base_dir}/results/${new_prefix}"
spm_dir="${data_dir}/spm"

TRAIN_SRC="${data_dir}/train.src"
TRAIN_TGT="${data_dir}/train.tgt"
DEV_SRC="${data_dir}/dev.src"
DEV_TGT="${data_dir}/dev.tgt"

rm -rf ${data_dir} ${checkpoints_dir} ${dict_dir} ${results_dir} ${spm_dir}
mkdir -p ${data_dir} ${checkpoints_dir} ${dict_dir} ${results_dir} ${spm_dir}
cp -r ${base_dir}/checkpoints/${prefix}/* ${checkpoints_dir}/
cp -r ${base_dir}/data/${prefix}/dict/* ${dict_dir}/
cp -r ${base_dir}/data/${prefix}/spm/${prefix}.model ${spm_dir}/${new_prefix}.model
cp -r ${base_dir}/data/${prefix}/spm/${prefix}.vocab ${spm_dir}/${new_prefix}.vocab

rm -rf ${TRAIN_SRC} ${TRAIN_TGT} ${MONO_SRC} ${MONO_TGT} ${DEV_SRC} ${DEV_TGT}
touch ${TRAIN_SRC} ${TRAIN_TGT} ${MONO_SRC} ${MONO_TGT} ${DEV_SRC} ${DEV_TGT}

SRC_LANG="en_XX"
SRC_LANG_SHORT="${SRC_LANG:0:2}"
SRC_LANG_CAPS="${SRC_LANG_SHORT^^}"
for TGT_LANG in "${test_langs[@]}"
do
    TGT_LANG_SHORT="${TGT_LANG:0:2}"
    TGT_LANG_CAPS="${TGT_LANG_SHORT^^}"
    PARALLEL_DIR="${datasets_dir}/en-${TGT_LANG_SHORT}-uralic"
    
    TRAIN_PREFIX="${PARALLEL_DIR}/train"
    DEV_PREFIX="${PARALLEL_DIR}/dev"
    TRAIN_NEW="${PARALLEL_DIR}/preprocessed/${new_prefix}/train"
    DEV_NEW="${PARALLEL_DIR}/preprocessed/${new_prefix}/dev"
    rm -rf ${PARALLEL_DIR}/preprocessed/${new_prefix}
    mkdir -p ${PARALLEL_DIR}/preprocessed/${new_prefix}

    ${SPM}/spm_encode --model=${SPM_MODEL} < ${TRAIN_PREFIX}.${SRC_LANG} > ${TRAIN_NEW}.spm.${SRC_LANG_SHORT}
    ${SPM}/spm_encode --model=${SPM_MODEL} < ${TRAIN_PREFIX}.${TGT_LANG} > ${TRAIN_NEW}.spm.${TGT_LANG_SHORT}

    ${SPM}/spm_encode --model=${SPM_MODEL} < ${DEV_PREFIX}.${SRC_LANG} > ${DEV_NEW}.spm.${SRC_LANG_SHORT}
    ${SPM}/spm_encode --model=${SPM_MODEL} < ${DEV_PREFIX}.${TGT_LANG} > ${DEV_NEW}.spm.${TGT_LANG_SHORT}

    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' ${TRAIN_NEW}.spm.${SRC_LANG_SHORT} >> ${TRAIN_SRC}
    awk -v TGT=$TGT_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } ' ${TRAIN_NEW}.spm.${TGT_LANG_SHORT} >> ${TRAIN_TGT}

    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' ${TRAIN_NEW}.spm.${SRC_LANG_SHORT} >> ${TRAIN_TGT}
    awk -v TGT=$TGT_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } ' ${TRAIN_NEW}.spm.${TGT_LANG_SHORT} >> ${TRAIN_SRC}    

    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' ${DEV_NEW}.spm.${SRC_LANG_SHORT} >> ${DEV_SRC}
    awk -v TGT=$TGT_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } ' ${DEV_NEW}.spm.${TGT_LANG_SHORT} >> ${DEV_TGT}

    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' ${DEV_NEW}.spm.${SRC_LANG_SHORT} >> ${DEV_TGT}
    awk -v TGT=$TGT_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } ' ${DEV_NEW}.spm.${TGT_LANG_SHORT} >> ${DEV_SRC}


done

python ${FAIRSEQ}/preprocess.py \
    --source-lang src --target-lang tgt \
    --trainpref "${data_dir}/train" \
    --validpref "${data_dir}/dev" \
    --srcdict ${dict_dir}/dict.src.txt \
    --tgtdict ${dict_dir}/dict.src.txt \
    --destdir ${data_dir} \
    --thresholdtgt 0 --thresholdsrc 0 --seed 0 \
    --amp --workers 42

cd "${base_dir}/src/${new_prefix}"
sbatch mlft.sh
cd -

# Preprocessing to launch BLFT

SRC_LANG="en_XX"
SRC_LANG_SHORT="${SRC_LANG:0:2}"
SRC_LANG_CAPS="${SRC_LANG_SHORT^^}"
for TGT_LANG in "${test_langs[@]}"
do
    TGT_LANG_SHORT="${TGT_LANG:0:2}"
    TGT_LANG_CAPS="${TGT_LANG_SHORT^^}"
    PARALLEL_DIR="${datasets_dir}/en-${TGT_LANG_SHORT}-uralic"
    new_prefix="sample-blft_${SRC_LANG_SHORT}-${TGT_LANG_SHORT}"

    data_dir="${base_dir}/data/${new_prefix}"
    checkpoints_dir="${base_dir}/checkpoints/${new_prefix}"
    dict_dir="${data_dir}/dict"
    results_dir="${base_dir}/results/${new_prefix}"
    spm_dir="${data_dir}/spm"

    rm -rf ${data_dir} ${checkpoints_dir} ${dict_dir} ${results_dir} ${spm_dir}
    mkdir -p ${data_dir} ${checkpoints_dir} ${dict_dir} ${results_dir} ${spm_dir}
    cp -r ${base_dir}/checkpoints/${prefix}/* ${checkpoints_dir}/
    cp -r ${base_dir}/data/${prefix}/dict/* ${dict_dir}/
    cp -r ${base_dir}/data/${prefix}/spm/${prefix}.model ${spm_dir}/${new_prefix}.model
    cp -r ${base_dir}/data/${prefix}/spm/${prefix}.vocab ${spm_dir}/${new_prefix}.vocab
    
    TRAIN_PREFIX="${PARALLEL_DIR}/train"
    DEV_PREFIX="${PARALLEL_DIR}/dev"
    TRAIN_NEW="${PARALLEL_DIR}/preprocessed/${new_prefix}/train"
    DEV_NEW="${PARALLEL_DIR}/preprocessed/${new_prefix}/dev"
    rm -rf ${PARALLEL_DIR}/preprocessed/${new_prefix}
    mkdir -p ${PARALLEL_DIR}/preprocessed/${new_prefix}

    ${SPM}/spm_encode --model=${SPM_MODEL} < ${TRAIN_PREFIX}.${SRC_LANG} > ${TRAIN_NEW}.spm.${SRC_LANG_SHORT}
    ${SPM}/spm_encode --model=${SPM_MODEL} < ${TRAIN_PREFIX}.${TGT_LANG} > ${TRAIN_NEW}.spm.${TGT_LANG_SHORT}

    ${SPM}/spm_encode --model=${SPM_MODEL} < ${DEV_PREFIX}.${SRC_LANG} > ${DEV_NEW}.spm.${SRC_LANG_SHORT}
    ${SPM}/spm_encode --model=${SPM_MODEL} < ${DEV_PREFIX}.${TGT_LANG} > ${DEV_NEW}.spm.${TGT_LANG_SHORT}

    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' ${TRAIN_NEW}.spm.${SRC_LANG_SHORT} > ${TRAIN_NEW}.prefixed.${SRC_LANG_SHORT}
    awk -v TGT=$TGT_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } ' ${TRAIN_NEW}.spm.${TGT_LANG_SHORT} > ${TRAIN_NEW}.prefixed.${TGT_LANG_SHORT}

    awk -v SRC=$SRC_LANG_CAPS '{print "LANG_TOK_"SRC" " $0 } ' ${DEV_NEW}.spm.${SRC_LANG_SHORT} > ${DEV_NEW}.prefixed.${SRC_LANG_SHORT}
    awk -v TGT=$TGT_LANG_CAPS '{print "LANG_TOK_"TGT" " $0 } ' ${DEV_NEW}.spm.${TGT_LANG_SHORT} > ${DEV_NEW}.prefixed.${TGT_LANG_SHORT}

    python ${FAIRSEQ}/preprocess.py \
        --source-lang ${SRC_LANG_SHORT} --target-lang ${TGT_LANG_SHORT} \
        --trainpref ${TRAIN_NEW}.prefixed \
        --validpref ${DEV_NEW}.prefixed \
        --srcdict ${dict_dir}/dict.src.txt \
        --tgtdict ${dict_dir}/dict.src.txt \
        --destdir ${data_dir} \
        --thresholdtgt 0 --thresholdsrc 0 --seed 0 \
        --amp --workers 42

    cd "${base_dir}/src/sample-blft"
    sbatch --export=src_long=$SRC_LANG,tgt_long=$TGT_LANG train-mCASP.sh
    cd -

    new_prefix="sample-blft_${TGT_LANG_SHORT}-${SRC_LANG_SHORT}"

    data_dir="${base_dir}/data/${new_prefix}"
    checkpoints_dir="${base_dir}/checkpoints/${new_prefix}"
    dict_dir="${data_dir}/dict"
    results_dir="${base_dir}/results/${new_prefix}"
    spm_dir="${data_dir}/spm"

    rm -rf ${data_dir} ${checkpoints_dir} ${dict_dir} ${results_dir} ${spm_dir}
    mkdir -p ${data_dir} ${checkpoints_dir} ${dict_dir} ${results_dir} ${spm_dir}
    cp -r ${base_dir}/checkpoints/${prefix}/* ${checkpoints_dir}/
    cp -r ${base_dir}/data/${prefix}/dict/* ${dict_dir}/
    cp -r ${base_dir}/data/${prefix}/spm/${prefix}.model ${spm_dir}/${new_prefix}.model
    cp -r ${base_dir}/data/${prefix}/spm/${prefix}.vocab ${spm_dir}/${new_prefix}.vocab
    
    python ${FAIRSEQ}/preprocess.py \
        --source-lang ${TGT_LANG_SHORT} --target-lang ${SRC_LANG_SHORT} \
        --trainpref ${TRAIN_NEW}.prefixed \
        --validpref ${DEV_NEW}.prefixed \
        --srcdict ${dict_dir}/dict.src.txt \
        --tgtdict ${dict_dir}/dict.src.txt \
        --destdir ${data_dir} \
        --thresholdtgt 0 --thresholdsrc 0 --seed 0 \
        --amp --workers 42
    
    cd "${base_dir}/src/sample-blft"
    sbatch --export=src_long=$TGT_LANG,tgt_long=$SRC_LANG train-mCASP.sh
    cd -
done



